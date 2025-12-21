#include <aocl_dlp.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aocl {
// API: https://amd.github.io/aocl-dlp/api/gemm/

// qq herper c++...
// les types C++ disponible => limite les versions de compilateur GCC>13
using bfloat16_t = __bf16;
using float32_t  = float;

static constexpr int GET_TENSOR_ALIGNMENT() {
#if defined(__AVX512F__)
//  Align matrix buffers to cache line boundaries (64-byte alignment) ...
    return 64;
#else // AVX2
    return 32;
#endif
}

namespace {
// some intrisic helper.
#if defined(__AVX512BF16__)
static __inline__ void _mm512_storeu_pbh(void *P, __m512bh A) {
    _mm512_storeu_epi16(P, reinterpret_cast<__m512i>(A));
}
static __inline__ void _mm256_storeu_pbh(void *P, __m256bh A) {
    _mm256_storeu_epi16(P, reinterpret_cast<__m256i>(A));
}
static __inline__ void _mm_storeu_pbh(void *P, __m128bh A) {
    _mm_storeu_epi16(P, reinterpret_cast<__m128i>(A));
}
#else
#pragma omp declare simd linear
static inline bfloat16_t compute_fp32_to_bf16(float s) {
    union {
        float f;
        uint32_t i;
    } u;
    union {
        bfloat16_t f;
        uint16_t i;
    } u2;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        u2.i = (u.i >> 16) | 64; /* force to quiet */
    } else {
        u2.i = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    }
    return u2.f;
}
#endif

}

// les fct de conversion...
static inline void convert(const float* A, bfloat16_t* B, const std::size_t K) {
#if defined(__AVX512BF16__)
        auto Kfin = K%32;
        auto K32 = K - Kfin;
        // num_threads(2...)  proc_bind(?) schedule(static, K/4)
        // #       pragma omp parallel for [...]
        for (std::size_t k=0; k<K32; k+=32) {
            auto A0 = _mm512_loadu_ps(A+k   );
            auto A1 = _mm512_loadu_ps(A+k+16);
            auto B01 = _mm512_cvtne2ps_pbh(A1,A0);
            _mm512_storeu_pbh(B+k, B01);
        }
        if (Kfin > 0) {
            auto k = K32;
            if ( (K-k) >= 16) {
                auto A0 = _mm512_loadu_ps(A+k);
                auto B0 = _mm512_cvtneps_pbh(A0);
                _mm256_storeu_pbh(B+k, B0);
                k+=16;
            }
            if ( (K-k) >= 8) {
                auto A0 = _mm256_loadu_ps(A+k);
                auto B0 = _mm256_cvtneps_pbh(A0);
                _mm_storeu_pbh(B+k, B0);
                k+=8;
            }
            // on doit pouvoir faire cas 4 avec store masqué.
            for (; k<K; ++k) {
                union {
                    __bf16 in;
                    bfloat16_t out;
                } u;
                u.in = _mm_cvtness_sbh(A[k]);
                B[k] = u.out;
            }
        }
#else
        // convert without intrisic => OMP SIMD
#       pragma omp for simd
        for (std::size_t k=0; k<K; ++k) {
            B[k] = compute_fp32_to_bf16(A[k]);
        }
#endif
    }

// GGML_OP_MUL_MAT => compute Ct = Bt@A
template<bool PACK=false>
void mul_mat(const float32_t* A, const float32_t* B, float32_t* C,
             std::size_t M, std::size_t N, std::size_t K,
             std::size_t lda, std::size_t ldb, std::size_t ldc
) {
    aocl_gemm_f32f32f32of32('R', 'N', 'T',
                             N, M, K,
                             1.0f,
                             B, ldb, 'N',
                             A, lda, PACK?'R':'N',
                             0.0f,
                             C, ldc,
                             NULL // No post-operations
                            );
}
template<bool PACK=false>
void mul_mat_batch(std::vector<const float32_t*> A, std::vector<const float32_t*> B, std::vector<float32_t*> C,
             const std::size_t M, const std::size_t N, const std::size_t K,
             const std::size_t lda, const std::size_t ldb, const std::size_t ldc
) {
    // - https://amd.github.io/aocl-dlp/api/gemm/index.html#batch-gemm-operations
    static constexpr char _R = 'R';
    static constexpr char _N = 'N';
    static constexpr char _T = 'T';
    static constexpr float alfa = 1;
    static constexpr float beta = 0;
    static           dlp_metadata_t* metadata = nullptr;  // No post-operations
    const md_t size = C.size();
    const md_t M_v = (md_t)M;
    const md_t N_v = (md_t)N;
    const md_t K_v = (md_t)K;
    const md_t lda_v = (md_t)lda;
    const md_t ldb_v = (md_t)ldb;
    const md_t ldc_v = (md_t)ldc;
    aocl_batch_gemm_f32f32f32of32(&_R, &_N, &_T,
                             &N_v, &M_v, &K_v,
                             &alfa,
                             B.data(), &ldb_v,
                             A.data(), &lda_v,
                             &beta,
                             C.data(), &ldc_v,
                             1, &size,
                             &_N, PACK?&_R:&_N,
                             &metadata
                            );
}
static inline std::size_t get_reorder_size_fp32fp32fp32(std::size_t M, std::size_t K) {
    return ((aocl_get_reorder_buf_size_f32f32f32of32('R', 'T', 'B', K, M, nullptr)-1)/GET_TENSOR_ALIGNMENT()+1)*GET_TENSOR_ALIGNMENT();
}

template<bool PACK=false>
void mul_mat(const bfloat16_t* A, const bfloat16_t* B, float32_t* C,
             std::size_t M, std::size_t N, std::size_t K,
             std::size_t lda, std::size_t ldb, std::size_t ldc
) {
    aocl_gemm_bf16bf16f32of32('R', 'N', 'T',
                             N, M, K,
                             1.0f,
                             (const bfloat16*) (const void*)B, ldb, 'N',
                             (const bfloat16*) (const void*)A, lda, PACK?'R':'N',
                             0.0f,
                             C, ldc,
                             NULL // No post-operations
                            );
}
template<bool PACK=false>
void mul_mat_batch(std::vector<const bfloat16_t*> A, std::vector<const bfloat16_t*> B, std::vector<float32_t*> C,
             const std::size_t M, const std::size_t N, const std::size_t K,
             const std::size_t lda, const std::size_t ldb, const std::size_t ldc
) {
    // - https://amd.github.io/aocl-dlp/api/gemm/index.html#batch-gemm-operations
    static constexpr char _R = 'R';
    static constexpr char _N = 'N';
    static constexpr char _T = 'T';
    static constexpr float alfa = 1;
    static constexpr float beta = 0;
    static           dlp_metadata_t* metadata = nullptr;  // No post-operations
    const md_t size = C.size();
    const md_t M_v = (md_t)M;
    const md_t N_v = (md_t)N;
    const md_t K_v = (md_t)K;
    const md_t lda_v = (md_t)lda;
    const md_t ldb_v = (md_t)ldb;
    const md_t ldc_v = (md_t)ldc;
    aocl_batch_gemm_bf16bf16f32of32(&_R, &_N, &_T,
                             &N_v, &M_v, &K_v,
                             &alfa,
                             (const bfloat16**) (const void**)B.data(), &ldb_v,
                             (const bfloat16**) (const void**)A.data(), &lda_v,
                             &beta,
                             C.data(), &ldc_v,
                             1, &size,
                             &_N, PACK?&_R:&_N,
                             &metadata
                            );
}
static inline std::size_t get_reorder_size_bf16bf16fp32(std::size_t M, std::size_t K) {
    return ((aocl_get_reorder_buf_size_bf16bf16f32of32('R', 'T', 'B', K, M, nullptr)-1)/GET_TENSOR_ALIGNMENT()+1)*GET_TENSOR_ALIGNMENT();
}

// TODO: add merger OP mulmat_add mulmat_bias... 
// TODO: "simple" OP
}
