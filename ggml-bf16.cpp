#include "ggml-bf16.h"
#include "ggml-backend-impl.h"

#include <iostream>

// => @ mettre dans un .h d'implementation!
struct _bf16_t {
  unsigned short u=0x8000; // +0?
};

using fp32_t = float;
using bf16_t = struct _bf16_t;

/*

make LLAMA_BF16=1 -j16
OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./llama-cli -m ~/LLM/llamafiles/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct.BF16.gguf -n 16 -t 0 -p "bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN"
OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./llama-cli -m ~/LLM/llamafiles/mistral-7b-instruct-v0.2/mistral-7b-instruct-v0.2.BF16.gguf -n 17 -t 0 -p "[INST]bonjour a tu un nom. je ne sais pas comment t'appeler. Si tu n'en as pas je peux t'appeler TINTIN[/INST]"

OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./llama-bench -m ~/LLM/llamafiles/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct.BF16.gguf -n 16 -p "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,126,256,512,1024,2048" -r 3
OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./llama-bench -m ~/LLM/llamafiles/mistral-7b-instruct-v0.2/mistral-7b-instruct-v0.2.BF16.gguf -n 16 -p "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,126,256,512,1024,2048" -r 3
    
*/

static inline std::ostream& operator<<(std::ostream& os, enum ggml_type type) {
    return os << ggml_type_name(type);
}

static inline std::ostream& operator<<(std::ostream& os, enum ggml_op type) {
    return os << ggml_op_name(type);
}

static inline std::ostream& operator<<(std::ostream& os, const struct ggml_tensor * t) {
    os <<"["<<t->ne[0];
    for (int i=1; i<GGML_MAX_DIMS ; i++) {
        if(t->ne[i]!=1) os <<":"<<t->ne[i];
    }
    return os <<"]@"<< t->type;
}

struct log_srcs {
    log_srcs(const ggml_tensor * t): _t(t){}
    const ggml_tensor * _t;
};

//static inline std::ostream& log_srcs(std::ostream& os, const struct ggml_tensor * t) {
static inline std::ostream& operator<<(std::ostream& os, const struct log_srcs t0) {
    auto t = t0._t;
    if (t->src[0]) os<<"s0"<<t->src[0];
    for(int i=1; i<GGML_MAX_SRC; i++) {
        if (t->src[i]) os<<", s"<<i<<t->src[i];
    }
    return os;
}

template<typename T> inline bool is(const struct ggml_tensor * t) {return false;}
template<> inline bool is<fp32_t>(const struct ggml_tensor * t) {return t->type==GGML_TYPE_F32;}
template<> inline bool is<bf16_t>(const struct ggml_tensor * t) {return t->type==GGML_TYPE_BF16;}

template<bool RUN> struct type {};
template<> struct type<false>{
    using R = bool;
    using T = const struct ggml_tensor *;
};
template<> struct type<true>{
    using R = void;
    using T = struct ggml_tensor *;
};
    
static inline void* add_byte(void* addr, size_t nb) {
    return (void*) (((char*)addr)+nb);
}

template<typename T>
class Matrice {
public:
    inline auto DIM1() const { return m_m; }
    inline auto DIM2() const { return m_n; }
    inline auto LD() const { return m_l; }

    const std::size_t m_m;  // m contigue
    const std::size_t m_n;  
    const std::size_t m_l;  // nb elements pour passer a la colonne suivante.
    T* m_values;
    
    inline Matrice(T* v, std::size_t n, std::size_t m, std::size_t l): m_n(n), m_m(m), m_l(l), m_values(v) {} 
    
    //                mul_mat(Matrice<bf16_t>(src0,src0->ne[0],src0->ne[1],src0->nb[2]/src0->nb[1]));
    inline Matrice(struct ggml_tensor * t): m_m(t->ne[0]), m_n(t->ne[1]), m_l(t->nb[1]/t->nb[0]), m_values((T*)(t->data)) {
        GGML_ASSERT(t->ne[2]==1);
        GGML_ASSERT(t->ne[3]==1);
    }

    inline Matrice(struct ggml_tensor * t, std::size_t i, std::size_t j): m_m(t->ne[0]), m_n(t->ne[1]), m_l(t->nb[1]/t->nb[0]) { //, m_values((T*)add_byte(t->data,)) {
        m_values = (T*) add_byte(t->data, i*t->nb[2]+j*t->nb[3]);
    }

    inline T& operator()(size_t i, size_t j) {
        return m_values[j*m_l+i];
    }
    inline const T operator()(size_t i, size_t j) const {
        return m_values[j*m_l+i];
    }

    inline T* addr(size_t i, size_t j) {
        return m_values+j*m_l+i;
    }
    inline const T* addr(size_t i, size_t j) const {
        return m_values+j*m_l+i;
    }

    static bool valid(const struct ggml_tensor * t) {
        return ggml_is_contiguous(t) && is<T>(t); // && t->ne[2]==1 && t->ne[3]==1;
    }
};

class ggml_backend_bf16_context {
public:
    const char * name() { return "BF16"; }
    
    ggml_backend_buffer_type_t get_default_buffer_type() { return ggml_backend_cpu_buffer_type(); }

    enum ggml_status graph_compute(struct ggml_cgraph * cgraph) {
        //  > MUL_MAT( [14336:4096]@bf16, s1 [14336:7]@f32 =>  [4096:7]@f32): ffn_out-29
        // => src0 = poids src1 = In => DST = T(SRC0) * SRC1 ...
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];

            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    //std::cout << " > " <<node->op<<"("<<log_srcs(node)<<" => "<<node<<"): "<< node->name<<std::endl;
                    mul_mat<true>(node);
                    break;

                //case GGML_OP_OUT_PROD:
                //    ggml_backend_blas_out_prod(ctx, node);
                //    break;

                // y a ca dans backend OPENBLAS ... sais pas pourquoi.
                case GGML_OP_NONE:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                    break;

                default:
                    fprintf(stderr, "%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                    GGML_ASSERT(false);
            }
        }

        return GGML_STATUS_SUCCESS;
    }

    bool supports_op(const struct ggml_tensor * op) {
    #if defined(__AVX512BF16__)
        if (op->op == GGML_OP_MUL_MAT) {
            if (mul_mat<false>(op)) {
                return true;
            }
        //} else if (op->op == GGML_OP_OUT_PROD) {
        //    std::cout << " > " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name<<std::endl;
        }
    #endif
        //std::cout << " > " <<op->op<<"("<<log_srcs(op)<<" => "<<op<<"): "<< op->name<<std::endl;
        return false;
    }
    
    bool supports_buft(ggml_backend_buffer_type_t buft) {
        return ggml_backend_buft_is_host(buft);
    }
    
private:

    // GGML_OP_MUL_MAT.
    template<bool RUN> // 
    //inline auto mul_mat(typename type<RUN>::t<struct ggml_tensor>::T op) -> decltype(type<RUN>::fct() {
    inline auto mul_mat(typename type<RUN>::T op) -> typename type<RUN>::R {
        const auto src0 = op->src[0];
        const auto src1 = op->src[1];
        auto dst  = op;
        
        if( Matrice<bf16_t>::valid(src0) &&
            Matrice<fp32_t>::valid(src1) &&
            Matrice<fp32_t>::valid(op)   &&
            src0->ne[0] % 32 == 0 // K%32==0 => pour l'instant pas d'autre cas... mais c'est (presque) tjs le cas!
            ) {
            if constexpr (RUN) {
                // broadcast factors
                const auto r2 = src1->ne[2]/src0->ne[2];
                const auto r3 = src1->ne[3]/src0->ne[3];
            
                for (int64_t i13 = 0; i13 < src1->ne[3]; i13++) {
                for (int64_t i12 = 0; i12 < src1->ne[2]; i12++) {
                    const auto i03 = i13/r3;
                    const auto i02 = i12/r2;

                    const Matrice<bf16_t> A(src0,i02,i03);
                    const Matrice<fp32_t> B(src1,i12,i13);
                          Matrice<fp32_t> C(dst,i12,i13);
                    mul_mat(A, B, C);
                }}
            } else {
                return true;
            }
        }
        if constexpr (!RUN) {
            return false;
        }
    }
    //  - bf16+fp32=fp32
    void mul_mat(const Matrice<bf16_t>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C);
    //  - autres cas ?
};

////////////////////////////////////////
// les methodes du backend => wrapper (elle sont codé dans la classe)
//faire ca avec des template ?
// - https://stackoverflow.com/questions/9281172/how-do-i-write-a-pointer-to-member-function-with-stdfunction
// - https://stackoverflow.com/questions/65778607/deduce-template-arguments-from-stdfunction-parameter-types
// ca doit pouvoir marcher...

GGML_CALL static const char * ggml_backend_bf16_name(ggml_backend_t backend) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    return ctx->name();
}

GGML_CALL static void ggml_backend_bf16_free(ggml_backend_t backend) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    delete ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_bf16_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    return ctx->get_default_buffer_type();
}

GGML_CALL static enum ggml_status ggml_backend_bf16_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    return ctx->graph_compute(cgraph);
}

GGML_CALL static bool ggml_backend_bf16_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    return ctx->supports_op(op);
}

GGML_CALL static bool ggml_backend_bf16_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    ggml_backend_bf16_context * ctx = (ggml_backend_bf16_context *)backend->context;
    return ctx->supports_buft(buft);
}

////////////////////////////////////////
// l'init du backend
static struct ggml_backend_i blas_backend_i = {
    /* .get_name                = */ ggml_backend_bf16_name,  // wraper(ggml_backend_bf16_context::name)
    /* .free                    = */ ggml_backend_bf16_free,
    /* .get_default_buffer_type = */ ggml_backend_bf16_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_bf16_graph_compute,
    /* .supports_op             = */ ggml_backend_bf16_supports_op,
    /* .supports_buft           = */ ggml_backend_bf16_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_bf16_guid(void) {
    static ggml_guid guid = { 0xca, 0xfe, 0xde, 0xca, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };
    return &guid;
}

ggml_backend_t ggml_backend_bf16_init(void) {
    // voir quel contrainte mettre ... et est-ce que ca doit etre ici?
    ggml_backend_bf16_context * ctx = new ggml_backend_bf16_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_bf16_guid(),
        /* .interface = */ blas_backend_i,
        /* .context   = */ ctx,
    };

    return backend;
}


//
/*
  => couche Meta-Llama-3-8B-Instruct CPU (?)
 > RMS_NORM(s0[4096:7]@f32 => [4096:7]@f32): norm-23
 > MUL(s0[4096:7]@f32, s1[4096]@f32 => [4096:7]@f32): attn_norm-23
 > MUL_MAT(s0[4096:4096]@bf16, s1[4096:7]@f32 => [4096:7]@f32): Qcur-23
 > ROPE(s0[128:32:7]@f32, s1[7]@i32 => [128:32:7]@f32): Qcur-23
 > MUL_MAT(s0[4096:1024]@bf16, s1[4096:7]@f32 => [1024:7]@f32): Kcur-23
 > ROPE(s0[128:8:7]@f32, s1[7]@i32 => [128:8:7]@f32): Kcur-23
 > MUL_MAT(s0[4096:1024]@bf16, s1[4096:7]@f32 => [1024:7]@f32): Vcur-23
 > CPY(s0[128:8:7]@f32, s1[7168]@f16 => [7168]@f16): k_cache_view-23 (copy of Kcur-23)
 > CPY(s0[7:1024]@f32, s1[7:1024]@f16 => [7:1024]@f16): v_cache_view-23 (copy of Vcur-23 (transposed))
?> MUL_MAT(s0[128:32:8]@f16, s1[128:7:32]@f32 => [32:7:32]@f32): kq-23
 > SOFT_MAX(s0[32:7:32]@f32, s1[32:32]@f32 => [32:7:32]@f32): kq_soft_max_ext-23
?> MUL_MAT(s0[32:128:8]@f16, s1[32:7:32]@f32 => [128:7:32]@f32): kqv-23
 > CONT(s0[128:32:7]@f32 => [4096:7]@f32): kqv_merged_cont-23
 > MUL_MAT(s0[4096:4096]@bf16, s1[4096:7]@f32 => [4096:7]@f32): kqv_out-23
 > ADD(s0[4096:7]@f32, s1[4096:7]@f32 => [4096:7]@f32): ffn_inp-23
 > RMS_NORM(s0[4096:7]@f32 => [4096:7]@f32): norm-23
 > MUL(s0[4096:7]@f32, s1[4096]@f32 => [4096:7]@f32): ffn_norm-23
 > MUL_MAT(s0[4096:14336]@bf16, s1[4096:7]@f32 => [14336:7]@f32): ffn_gate-23
 > UNARY(s0[14336:7]@f32 => [14336:7]@f32): ffn_silu-23
 > MUL_MAT(s0[4096:14336]@bf16, s1[4096:7]@f32 => [14336:7]@f32): ffn_up-23
 > MUL(s0[14336:7]@f32, s1[14336:7]@f32 => [14336:7]@f32): ffn_gate_par-23
 > MUL_MAT(s0[14336:4096]@bf16, s1[14336:7]@f32 => [4096:7]@f32): ffn_out-23
 > ADD(s0[4096:7]@f32, s1[4096:7]@f32 => [4096:7]@f32): l_out-23
*/

//////////////////////////////////////////////////////////////
#include <immintrin.h>

// cas K%32 = 0:
inline auto load(const fp32_t *X) {
    auto x1 = _mm512_loadu_ps(X);
    auto x2 = _mm512_loadu_ps(X+16);
    return _mm512_cvtne2ps_pbh(x2,x1);
}
inline auto load(const bf16_t *X) {
    return (__m512bh) _mm512_loadu_epi16(X);
}

inline auto madd(const __m512bh& A, const __m512bh& B, const __m512& C) {
    return _mm512_dpbf16_ps(C, A, B);
}

inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}

inline void store(bf16_t *pX, const __m512bh& x) {
    _mm512_storeu_epi16(pX, (__m512i)x);
}

// write C after last reduction
template<typename... T>
inline void store(fp32_t *pX, T&&... x) {
    constexpr __mmask16 _m = ((1<<sizeof...(T))-1);
    auto pack = hadd(std::forward<T>(x)...);
    _mm512_mask_storeu_ps(pX, _m, pack);
}

// p'etre un "masque" A(quantisé>bf16)/B(fp32>bf16)
enum class ACTION {
    NONE,
    STORE,
    LOAD
    // + ACC / C config!
};

template<size_t M, size_t N, ACTION ACT=ACTION::NONE, bool ACC=false, typename TA, typename TB, typename TC>
void gemm(const TA *pA, const TB *pB, TC *pC, std::size_t lda, std::size_t ldb, std::size_t ldc, std::size_t K, bf16_t *pB_=nullptr, std::size_t ldb_=0) {
    constexpr int K0 = 32; // 32 bf16 !
    static_assert(N>0);
    static_assert(M>0);
    // K%32 == 0!!
    // A[?,K+:lda]
    // B[?,K+:ldb]
    // C[?,ldc]
    __m512   C[M][N];
    __m512bh A[M];
    __m512bh B;
    #pragma GCC unroll 100
    for(size_t j=0; j<N; j++) {
        #pragma GCC unroll 100
        for(size_t i=0; i<M; i++) {
            C[i][j] = _mm512_setzero_ps();
        }
    }
    for (std::size_t k=0; k<K; k+=K0) {
        #pragma GCC unroll 100
        for(size_t i=0; i<M; i++) {
            A[i] = load(pA+i*lda+k);
        }
        #pragma GCC unroll 100
        for(size_t j=0; j<N; j++) {
            // gestion d'un cache pour B
            if constexpr(ACT!=ACTION::LOAD) B = load(pB+j*ldb+k);
            if constexpr(ACT==ACTION::LOAD) B = load(pB_+j*ldb_+k);
            #pragma GCC unroll 100
            for(size_t i=0; i<M; i++) {
                C[i][j] = madd(A[i], B, C[i][j]);
            }
            if constexpr(ACT==ACTION::STORE) store(pB_+j*ldb_+k, B);
        }
    }
        
    // reduce and store C res.
    #pragma GCC unroll 100
    for(size_t j=0; j<N; j++) {
        #pragma GCC unroll 100
        for(size_t i=0; i<M; i++) {
            if constexpr (ACC) {
                pC[i+j*ldc] += hsum(C[i][j]);
            } else {
                pC[i+j*ldc] = hsum(C[i][j]);
            }
        }
    }
}

template<size_t M, size_t N, ACTION ACT=ACTION::NONE, bool ACC=false, typename TA, typename TB, typename TC>
inline void sgemm_512_bloc(TA* A, TB* B, TC* C, size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc, bf16_t* B_, size_t ldb_) {
    GGML_ASSERT(m<=M);
    GGML_ASSERT(n<=N);

    // choix du kernel:
    if ((M==m) && (N==n)) { // seul cas traité pour l'instant
        gemm<M,N,ACT,ACC>(A, B, C, lda, ldb, ldc, k, B_,ldb_);
        return;
    }
    if constexpr (M>1) { // arret de la recursion
        if (M>m) {
            sgemm_512_bloc<M-1,N,ACT,ACC>(A,B,C,m,n,k,lda,ldb,ldc, B_,ldb_);
        }
    }
    if constexpr (N>1) { // arret de la recursion
        if (M==m && N>n) {
            sgemm_512_bloc<M,N-1,ACT,ACC>(A,B,C,m,n,k,lda,ldb,ldc, B_,ldb_);
        }
    }
}

template<size_t M1, size_t N1, size_t M0, size_t N0, size_t K0=1024, typename TA, typename TB, typename TC>
inline void sgemm_512_bloc(const Matrice<TA>& A, const Matrice<TB>& B, Matrice<TC>& C, size_t I0, size_t J0, bf16_t* B_) {
    const size_t IN = std::min(C.DIM1(), I0+M1*M0);
    const size_t JN = std::min(C.DIM2(), J0+N1*N0);
    const auto KN = A.DIM1(); // == B.DIM1()
    
    
    for (size_t k=0; k<KN; k+=K0) {
        const auto _K = std::min(K0,KN-k);
        for (size_t j=J0; j<JN; j+=N0) {
            const auto _N = std::min(N0,JN-j);
            if (k==0) {
                sgemm_512_bloc<M0,N0,ACTION::STORE,false>(A.addr(0,I0),B.addr(0,j),C.addr(I0,j),std::min(M0,IN-I0),_N,_K, A.LD(),B.LD(),C.LD(), B_, K0);
            } else {
                sgemm_512_bloc<M0,N0,ACTION::STORE,true>(A.addr(k,I0),B.addr(k,j),C.addr(I0,j),std::min(M0,IN-I0),_N,_K, A.LD(),B.LD(),C.LD(), B_, K0);
            }
            for (size_t i=I0+M0; i<IN; i+=M0) {
                const auto _M = std::min(M0,IN-i);
                if (k==0) {
                    sgemm_512_bloc<M0,N0,ACTION::LOAD,false>(A.addr(0,i),B.addr(0,j),C.addr(i,j),_M,_N,_K, A.LD(),B.LD(),C.LD(), B_, K0);
                } else {
                    sgemm_512_bloc<M0,N0,ACTION::LOAD,true>(A.addr(k,i),B.addr(k,j),C.addr(i,j),_M,_N,_K, A.LD(),B.LD(),C.LD(), B_, K0);
                }
            }
        }
    }    
}

void ggml_backend_bf16_context::mul_mat(const Matrice<bf16_t>& A, const Matrice<fp32_t>& B, Matrice<fp32_t>& C) {
    const auto m = C.DIM1(); // == A.DIM2()
    const auto n = C.DIM2(); // == B.DIM2()
    const auto k = A.DIM1(); // == B.DIM1()
    GGML_ASSERT(A.LD()>=k);
    GGML_ASSERT(B.LD()>=k);
    GGML_ASSERT(C.LD()>=m);
    if(n<=4) {
        constexpr size_t M0 = 6;
        constexpr size_t N0 = 4;
        constexpr size_t M1 = 8;
        constexpr size_t K0 = 4096;
        bf16_t B_cache[N0*K0];

        #pragma omp parallel for private(B_cache)
        for (size_t i=0; i<m; i+=M1*M0) {
            sgemm_512_bloc<M1,1,M0,N0,K0>(A, B, C, i, 0, B_cache);
        }
    } else {
        // la taille des plus grand blocs.
        constexpr size_t M0 = 5;
        constexpr size_t N0 = 5;
        constexpr size_t M1 = 8;
        constexpr size_t N1 = 4;
        constexpr size_t K0 = 4096;
        bf16_t B_cache[N0*K0];

        #pragma omp parallel for collapse(2) private(B_cache)
        for (size_t i=0; i<m; i+=M1*M0) {
            for (size_t j=0; j<n; j+=N1*N0) {
                sgemm_512_bloc<M1,N1,M0,N0,K0>(A, B, C, i, j, B_cache);
            }
        }
    }
}
/* ~/LLM/llamafiles/mistral-7b-instruct-v0.2/mistral-7b-instruct-v0.2.BF16.gguf ...
| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |           pp1 |      3.89 ± 0.03 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |           pp2 |      7.64 ± 0.05 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |           pp3 |     11.42 ± 0.33 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |           pp4 |     15.42 ± 0.07 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |           pp5 |     19.33 ± 0.09 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |           pp6 |     22.95 ± 0.13 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |           pp7 |     26.58 ± 0.19 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |           pp8 |     29.10 ± 1.18 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |           pp9 |     33.85 ± 0.23 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |          pp10 |     36.56 ± 0.65 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |          pp11 |     39.94 ± 0.10 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |          pp12 |     43.00 ± 0.47 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |          pp13 |     46.05 ± 0.47 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |          pp14 |     49.27 ± 0.40 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |          pp15 |     51.52 ± 0.25 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |          pp16 |     52.07 ± 0.46 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |          pp32 |     75.97 ± 0.50 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |          pp64 |     86.82 ± 0.22 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |         pp126 |     98.39 ± 0.87 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |         pp256 |    101.59 ± 0.73 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |         pp512 |     98.11 ± 0.22 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |        pp1024 |     93.71 ± 0.15 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |        pp2048 |     86.70 ± 0.06 |
| llama 7B BF16                  |  13.49 GiB |     7.24 B | CPU        |       8 |          tg16 |      3.93 ± 0.01 |
*/        

