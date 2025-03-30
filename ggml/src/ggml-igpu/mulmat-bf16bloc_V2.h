#pragma once
// @ builder avec HIP...

//#include "ggml-impl.h"
//#include "ggml-hip.h"
//#include "tools.h"

#include <iostream>

//#if 1
#if 0
#define __host__
#define __device__
#define __global__
#define __shared__
#define __launch_bounds__(...)
#endif

#include "types.h"
#include "tensor.h"

namespace ggml::backend::igpu {

    // les types vectoriels...
    using bfloat16x2_t  = __bf16 __attribute__((ext_vector_type(2)));
    using bfloat16x16_t = __bf16 __attribute__((ext_vector_type(16)));
    using float32x8_t   =  float __attribute__((ext_vector_type(8)));

    using hipint_t = int;  // std::size_t => trop de SGPRs!

    template<typename... ARGS>
    constexpr __device__ hipint_t HIP_MAX(const hipint_t A, const hipint_t B, const ARGS... args) {
        if constexpr (sizeof...(args) == 0) {
            return A>B?A:B;
        } else {
            return HIP_MAX(A>B?A:B, args...);
        }
    }

    constexpr int MILIEUX = 0x0;
    constexpr int DEBUT   = 0x1;
    constexpr int FIN     = 0x2;

    template<typename T> __device__ bfloat16_t conv2bf16(T val);
    template<> __device__ inline bfloat16_t conv2bf16(float32_t  val) { return val; }
    template<> __device__ inline bfloat16_t conv2bf16(bfloat16_t val) { return val; }

    //=====================================================================================
    // les kernels:
    //----------------------------------
    // - gemm:
    //   - reformat de B depuis le format ggml avec conversion
    template<hipint_t N1, hipint_t K1, hipint_t N0=16, hipint_t K0=8, hipint_t KV=2, typename TB>
    __global__ void __launch_bounds__(32) pack(const TB* __restrict__ b, bfloat16x2_t* __restrict__ b_cache, hipint_t N, hipint_t K) {
        const hipint_t tid = threadIdx.x;
        constexpr hipint_t NB_THREAD = 32; // blockDim.x

        const hipint_t j2 = blockIdx.x;
        const hipint_t J0 = j2*N0*N1; // + [0..N0*N1]

        __shared__ bfloat16x2_t B_trans[K1*K0][N0+1];
        // si seulement 32 thread pas besoin de __syncthreads()...
        for (hipint_t j1=0; j1<N1; ++j1) {
            // load
            const hipint_t Nx = min(N0, N - j1*N0 - j2*N0*N1);
            for (hipint_t id = tid; id<K1*K0*Nx; id+= NB_THREAD) {
                const hipint_t k0 = id%(K0*K1);
                const hipint_t j0 = id/(K0*K1);
                const hipint_t j = (j0+j1*N0+j2*N0*N1);
#               pragma unrool KV
                for (hipint_t kv=0; kv<KV; ++kv) {
                    B_trans[k0][j0][kv] = conv2bf16(b[k0*KV+kv + j*K]);
                }
            }
            if constexpr (NB_THREAD>32) __syncthreads();
            // store: @ limiter ?
            if ((J0+j1*N0) < N) {
                for (hipint_t id = tid; id<K1*K0*N0; id+= NB_THREAD) {
                    //                         j2       j1   k0    j0
                    // B_Cache: bfloat16x2_t[N/(N0*N1]][N1][K1*K0][N0];
                    const hipint_t j0 = id%N0;
                    const hipint_t k0 = id/N0;
                    b_cache[j0 + k0*N0 + j1*N0*K0*K1 + j2*N0*K0*K1*N1] = B_trans[k0][j0];
                }
            }
            if constexpr (NB_THREAD>32) __syncthreads();
        }
    }
    //   - calcul
    template< int TYPE,
    hipint_t M2, hipint_t N2, hipint_t K2,
    hipint_t M1, hipint_t N1, hipint_t K1,
    hipint_t M0=16, hipint_t N0=16, hipint_t K0=8, hipint_t KV=2,
    typename TA, typename TC>
    __global__ void __launch_bounds__(16*2*M2*N2) wmma_matmul(
            const TA* __restrict__ a, const bfloat16x2_t* __restrict__ b, TC* __restrict__ c,
            TC* __restrict__ c_cache,
            hipint_t M, hipint_t N, hipint_t K
    )
    {
        // only possible values!
        static_assert(     M0==16, "use of __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32");
        static_assert(     N0==16, "use of __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32");
        static_assert((KV*K0)==16, "use of __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32");

        using A_t = const bfloat16x2_t[M2][M1][K2*K1*K0][M0];
        using B_t = const bfloat16x2_t[N2][N1][K2*K1*K0][N0];

        //                                 [ NB_THREAD   ]
        using C_t = float32_t[N1][M1][M0/2][N2][M2][2][N0];

        const hipint_t tid = threadIdx.x + 32 * threadIdx.y + 32*M2*threadIdx.z;
        constexpr hipint_t NB_THREAD = 32*M2*N2;
        const hipint_t i2 = threadIdx.y;
        const hipint_t j2 = threadIdx.z;

        static_assert( (N0*K0*N1*N2*K1)%NB_THREAD == 0);
        static_assert( (M0*K0*M1*M2*K1)%NB_THREAD == 0);
        static_assert( (M0*M2*N0*N2)%NB_THREAD == 0);

        const hipint_t j3 = blockIdx.x;
        const hipint_t i3 = blockIdx.y;  // n° bloc sur M on a M/(M0*M1*M2)

        const hipint_t I0 = i2*M0*M1+ i3*M0*M1*M2; // + [0..N0*N1]
        const hipint_t J0 = j2*N0*N1+ j3*N0*N1*N2; // + [0..N0*N1]

        // pour les "calcul"
        const hipint_t i0 = threadIdx.x%M0;
        const hipint_t j0 = threadIdx.x%N0;

        A_t& __restrict__ A       = reinterpret_cast<A_t*> (a)[i3];
        B_t& __restrict__ B       = reinterpret_cast<B_t*> (b)[j3];
        C_t& __restrict__ C_cache = reinterpret_cast<C_t*> (c_cache)[i3+j3*(M/(M0*M1*M2))];

        // A => A[K/(K2*K1*K0*KV)][M/(M0*M1*M2)][M2][M1][K2][K1][K0][M0][KV]
        // en entrée A:  = bfloat16x2_t[K..][M/(M0*M1*M2)][M2][M1][K0][M0]; // KV=2
        // en entrée A:  = fp_E3M4x4_t [K..][M/(M0*M1*M2)][M2][M1][K0][M0]; // KV=4

        // pas de K on a un repack...
        // B => B                      [N/(N0*N1*N2)][N2][N1][K2][K1][K0][N0][2]
        // en entrée B:  = bfloat16x2_t[N/(N0*N1*N2)][N2][N1][K2][K1][K0][N0];

        // les shareds:
        using A_frag_t  = bfloat16x2_t[K1][M2][M1][K0][M0];
        using B_frag_t  = bfloat16x2_t[K1][N2][N1][K0][N0];
        using C_frag_t  = float32_t[N0][M2*M1*M0+1];

        __shared__ char data[HIP_MAX(sizeof(A_frag_t) + sizeof(B_frag_t), sizeof(C_frag_t))];
        A_frag_t*  A_frag  = reinterpret_cast<A_frag_t*> (&data[0]);
        B_frag_t*  B_frag  = reinterpret_cast<B_frag_t*> (&data[sizeof(A_frag_t)]);
        C_frag_t&  C_frag  = *reinterpret_cast<C_frag_t*> (&data[0]);

        // chargement de C (sauf debut)
        float32x8_t   c_frag[N1][M1] = {};  // [i0:i0+16] => [j0:j0+16 & i1:i1+M1 / thread]

        // calcul
        for (hipint_t k2=0; k2<K2; ++k2) {
            // chargement de A
            for (hipint_t id = tid; id<M0*K0*M1*M2*K1; id+=NB_THREAD) {
                const hipint_t i0 =  id%M0;
                const hipint_t k0 = (id/M0)%K0;
                const hipint_t k1 = (id/(M0*K0))%K1;
                const hipint_t i1 = (id/(M0*K0*K1))%M1;
                const hipint_t i2 =  id/(M0*K0*K1*M1);
                A_frag[0][k1][i2][i1][k0][i0] = A[i2][i1][k0+k1*K0+k2*K0*K1][i0];
            }
            // chargement de B
            for (hipint_t id = tid; id<N0*K0*K1*N1*N2; id+=NB_THREAD) {
                const hipint_t j0 =  id%N0;
                const hipint_t k0 = (id/N0)%K0;
                const hipint_t k1 = (id/(N0*K0))%K1;
                const hipint_t j1 = (id/(N0*K0*K1))%N1;
                const hipint_t j2 =  id/(N0*K0*K1*N1);
                if(j2*N0*N1+ j3*N0*N1*N2 < N) {
                    B_frag[0][k1][j2][j1][k0][j0] = B[j2][j1][k0+k1*K0+k2*K0*K1][j0];
                }
            }

            __syncthreads(); // OK il faut attendre que tout soit reformaté.

            if (J0<N) {
                // calcul:
                using a_frag_in_t  = bfloat16x2_t[K0];  // pour le load
                using a_frag_out_t = bfloat16x16_t;     // pour le calcul
                bfloat16_t a_frag[K0*KV];               // [M1xM0/thread]
                auto& a_frag_in  = reinterpret_cast<a_frag_in_t &>(a_frag);
                auto& a_frag_out = reinterpret_cast<a_frag_out_t&>(a_frag);

                using b_frag_in_t  = bfloat16x2_t[N1][K0];
                using b_frag_out_t = bfloat16x16_t[N1];
                bfloat16_t b_frag[N1][K0*KV];           // [N0/thread]  => copie sur M1
                auto& b_frag_in  = reinterpret_cast<b_frag_in_t &>(b_frag);
                auto& b_frag_out = reinterpret_cast<b_frag_out_t&>(b_frag);

                for (hipint_t k1=0; k1<K1; ++k1) {
                    // chargement de b_frag (le meme pour chaque i1)
                    for (hipint_t j1=0; j1<N1; ++j1) {
                        if ((J0+j1*N0)<N) {
#                           pragma unrool K0
                            for (int k0=0; k0<K0; k0++) {
                                b_frag_in[j1][k0] = B_frag[0][k1][j2][j1][k0][j0];
                            }
                        }
                    }

                    for (hipint_t i1=0; i1<M1; ++i1) {
                        // chargement de b_frag
#                       pragma unrool K0
                        for (int k0=0; k0<K0; k0++) {
                            a_frag_in[k0] = A_frag[0][k1][i2][i1][k0][i0];
                        }

                        for (hipint_t j1=0; j1<N1; ++j1) {
                            c_frag[j1][i1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag_out, b_frag_out[j1], c_frag[j1][i1]);
                        }
                    }
                }
            }
            __syncthreads(); // OK il faut attendre que tout soit reformaté.
        }

        // save C (sauf fin)
        if (J0<N) {
            for (hipint_t i1=0; i1<M1; ++i1) {
                for (hipint_t j1=0; j1<N1; ++j1) {
                    for (hipint_t id = tid%32; id<M0*N0 ; id+=32) {
                        const hipint_t j0   =  id%N0;
                        const hipint_t i0_a = (id/ N0)%2;
                        const hipint_t i0_b = (id/(N0*2))%(M0/2);
                        //  - debut
                        if constexpr(TYPE == DEBUT) {
                            C_cache[j1][i1][i0_b][j2][i2][i0_a][j0]  = c_frag[j1][i1][i0_b];
                        }
                        //  - milieux
                        if constexpr(TYPE == MILIEUX) {
                            C_cache[j1][i1][i0_b][j2][i2][i0_a][j0] += c_frag[j1][i1][i0_b];
                        }
                        //  - fin
                        if constexpr(TYPE == FIN) {
                            c_frag[j1][i1][i0_b] += C_cache[j1][i1][i0_b][j2][i2][i0_a][j0];
                        }
                        //  - fin&debut => rien c_frag est a jours
                    }
                }
            }
        }

        // ecriture C (si fin)
        if constexpr( (TYPE&FIN) == FIN) {
            for (hipint_t j2_=0; j2_<N2; ++j2_) {
                for (hipint_t j1=0; j1<N1; ++j1) {
                    if (j2 == j2_) {
                        // seulement ceux qui ont le bon...
                        for (hipint_t i1=0; i1<M1; ++i1) {
                            // on recupere 1 bloc...
                            for (hipint_t id = tid%32; id<M0*N0 ; id+=32) {
                                const hipint_t j0   =  id%N0;
                                const hipint_t i0_a = (id/ N0)%2;   //
                                const hipint_t i0_b = (id/(N0*2))%(M0/2);
                                const hipint_t i = i0_a+i0_b*2 + i1*M0+i2*M0*M1;
                                C_frag[j0][i] = c_frag[j1][i1][i0_b];
                            }
                        }
                    }
                    __syncthreads(); // OK il faut attendre que tout soit reformaté.

                    // shared => C!
                    const hipint_t Nx = min(N0, N-j1*N0-j2_*N0*N1-j3*N0*N1*N2);
                    for (hipint_t id = tid; id<Nx*M2*M1*M0 ; id+=NB_THREAD) {
                        const hipint_t i012 =  id%(M2*M1*M0);
                        const hipint_t j0   =  id/(M2*M1*M0);
                        const hipint_t i = i012                  + i3*M0*M1*M2;
                        const hipint_t j = j0   + j1*N0+j2_*N0*N1+ j3*N0*N1*N2;
                        c[i+j*M] = C_frag[j0][i012];
                    }
                    __syncthreads(); // OK il faut attendre que tout soit ecrit.
                }
            }
        }
    }

    //----------------------------------
    // - gemv:
    template< int TYPE,
    hipint_t M2, hipint_t K2,
    hipint_t M1, hipint_t K1,
    hipint_t M0=16, hipint_t N0=1, hipint_t K0=8, hipint_t KV=2,
    typename TA, typename TB, typename TC>
    __global__ void __launch_bounds__(16*2*M2) wmma_matvec(
            const TA* __restrict__ a, const TB* __restrict__ b, TC* __restrict__ c,
            hipint_t M, hipint_t K)
    {
        // only possible/interesting values!
        static_assert(     N0==1,  "compute mat@vect => N==1");
        static_assert(     M0==16, "use of __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32");
        static_assert((KV*K0)==16, "use of __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32");
        static_assert(   M1%2==0,  "pour optimiser la manipulation de C");

        using A_t = const bfloat16x2_t[M2][M1][K2*K1*K0][M0];
        using B_t = const bfloat16x2_t        [K2*K1*K0];

        constexpr hipint_t NB_THREAD = 32*M2;
        const hipint_t tid = threadIdx.x + 32 * threadIdx.y;
        const hipint_t i2 = threadIdx.y;
        const hipint_t i3 = blockIdx.x;

        // pour les "calcul"
        const hipint_t i0 = threadIdx.x%M0;

        A_t& __restrict__ A = reinterpret_cast<A_t*> (a)[i3];

        // les shareds:
        using B_frag_t  = bfloat16x2_t[K2][K1][K0];
        using C_frag_t  = float32_t[M2][M1][M0];

        __shared__ B_frag_t B_frag;
        __shared__ C_frag_t C_frag;

        // le calcul de C sur K0*K1*K2
        float32x8_t   c_frag[M1] = {};

        // chargement de B => shared
        for (hipint_t id = tid; id<K2*K1*K0; id+=NB_THREAD) {
            const hipint_t k0 = (id)%K0;
            const hipint_t k1 = (id/K0)%K1;
            const hipint_t k2 = (id/(K0*K1))%K2;
            for (hipint_t kv=0; kv<KV; ++kv) {
                B_frag[k2][k1][k0][kv] = conv2bf16(b[id*2+kv]);
            }
        }

        __syncthreads(); // OK il faut attendre que tout soit chargé.

        // calcul
        for (hipint_t k2=0; k2<K2; ++k2) {

            using a_frag_in_t  = bfloat16x2_t[K0];  // pour le load
            using a_frag_out_t = bfloat16x16_t;     // pour le calcul
            bfloat16_t a_frag[K0*KV];               // [M1xM0/thread]
            auto& a_frag_in  = reinterpret_cast<a_frag_in_t &>(a_frag);
            auto& a_frag_out = reinterpret_cast<a_frag_out_t&>(a_frag);

            using b_frag_in_t  = bfloat16x2_t[K0];
            using b_frag_out_t = bfloat16x16_t;
            bfloat16_t b_frag[K0*2];                // copie sur toutes les lignes
            auto& b_frag_in  = reinterpret_cast<b_frag_in_t &>(b_frag);
            auto& b_frag_out = reinterpret_cast<b_frag_out_t&>(b_frag);

            for (hipint_t k1=0; k1<K1; ++k1) {
                // chargement de b_frag (le meme pour chaque i1 & copier sur chaque lignes)
#               pragma unrool K0
                for (hipint_t k0=0; k0<K0; k0++) {
                    b_frag_in[k0] = B_frag[k2][k1][k0];
                }

                for (hipint_t i1=0; i1<M1; ++i1) {
#                   pragma unrool K0
                    for (hipint_t k0=0; k0<K0; k0++) {
                        a_frag_in[k0] =  A[i2][i1][k0+k1*K0+k2*K0*K1][i0];
                    }
                    c_frag[i1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag_out, b_frag_out, c_frag[i1]);
                }
            }
        }

        // relecture de C
        if constexpr((TYPE == MILIEUX) || (TYPE == FIN)) {
            for (hipint_t id = tid; id<M0*M1*M2 ; id+=NB_THREAD) {
                const hipint_t i0 =  id%M0;
                const hipint_t i1 = (id/M0)%M1;
                const hipint_t i2 = (id/(M0*M1));
                C_frag[i2][i1][i0] = c[id+i3*M0*M1*M2];
            }
            __syncthreads(); // OK il faut attendre que tout soit dans le shared.
        }
        // accumulation dans le shared
#       pragma unrool M1/2
        for (hipint_t i1_b=0; i1_b<M1; i1_b+=2) {
            const hipint_t i0_a = (tid/16)%2;  // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
            const hipint_t i0_b = (tid/2)%8;   // 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7
            const hipint_t i1_a =  tid%2;      // 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1
            if constexpr(TYPE == DEBUT) {
                C_frag[i2][i1_b+i1_a][i0_a + 2*i0_b] = c_frag[i1_b+i1_a][i0_b];
            }
            if constexpr(TYPE == (DEBUT|FIN)) {
                C_frag[i2][i1_b+i1_a][i0_a + 2*i0_b] = c_frag[i1_b+i1_a][i0_b];
            }
            if constexpr(TYPE == MILIEUX) {
                C_frag[i2][i1_b+i1_a][i0_a + 2*i0_b] += c_frag[i1_b+i1_a][i0_b];
            }
            if constexpr(TYPE == FIN) {
                C_frag[i2][i1_b+i1_a][i0_a + 2*i0_b] += c_frag[i1_b+i1_a][i0_b];
            }
        }
        __syncthreads(); // OK il faut attendre que tout soit dans le shared.
        // ecriture
        for (hipint_t id = tid; id<M0*M1*M2 ; id+=NB_THREAD) {
            const hipint_t i0 =  id%M0;
            const hipint_t i1 = (id/M0)%M1;
            const hipint_t i2 = (id/(M0*M1));
            c[id+i3*M0*M1*M2] = C_frag[i2][i1][i0];
        }
    }

    //=====================================================================================
    template<std::size_t N> constexpr std::size_t block_size(std::size_t size) { return (((size-1)/N) + 1) * N; }
    template<std::size_t N> constexpr std::size_t nb_block  (std::size_t size) { return (((size-1)/N) + 1); }

    template<typename T>
    class hip_cache {
        void clear() {
            if (m_size > 0) {
                ggml::hip::deallocateDevice(m_data);
                m_data = nullptr;
            }
            m_size = 0;
        }
    public:
        hip_cache() {}
        ~hip_cache() { clear(); }
        bool ensure_size(std::size_t size) {
            if (size > m_size) {
                clear();
                m_size = size;
                m_data = ggml::hip::allocateDevice<T>(m_size);
                return true;
            }
            return false;
        }
        T* m_data = nullptr; // not realy a T* ... it is a ref on device memory
    private:
        std::size_t m_size = 0;
    };

    // @ initialiser a l'init du backend en fonction de la taille des poids?
    //static hip_cache<bfloat16_t> B_cache;
    //static hip_cache<float32_t>  C_cache;

    //=====================================================================================
    // les fonctions hotes
    //----------------------------------
    // - gemm:
    template<std::size_t _M1, std::size_t _N1, std::size_t _K1, std::size_t _M2, std::size_t _N2, std::size_t _K2>
    void sgemm_wmma(const bfloat16_t* A, const float32_t* B, float32_t* C, std::size_t M, std::size_t N, std::size_t K) {
        static hip_cache<float32_t>     C_cache;
        static hip_cache<bfloat16x2_t>  B_cache;

        constexpr int M0 = 16;
        constexpr int N0 = 16;
        constexpr int KV = 2;
        constexpr int KB = 16;
        constexpr int K0 = KB/KV;

        constexpr int M1 = _M1;
        constexpr int N1 = _N1;
        constexpr int K1 = _K1;

        constexpr int M2 = _M2;
        constexpr int N2 = _N2;
        constexpr int K2 = _K2/(K1*KB);

        GGML_ASSERT((K%_K2) == 0);

        // les caches => voir a avoir un min pour N ~ 512?
        if (B_cache.ensure_size((_K2/KV) * block_size<N0*N1*N2>(N))) {
            IGPU_TRACE("B_cache[" << _K2 << ", " << block_size<N0*N1*N2>(N) <<"]");
        }
        if (C_cache.ensure_size(block_size<N0*N1*N2>(N) * block_size<M0*M1*M2>(M))) {
            IGPU_TRACE("C_cache[" << block_size<M0*M1*M2>(M) << ", " << block_size<N0*N1*N2>(N) <<"]");
        }

        if (K<=_K2) {
            // repack B
            hipLaunchKernelGGL(HIP_KERNEL_NAME(pack<N1*N2,_K2/KB>),
                    dim3(nb_block<N0*N1*N2>(N),1,1), dim3(32, 1, 1),
                    0, 0,
                    B,B_cache.m_data, N,K);
            auto res = hipGetLastError();
            if (res != hipSuccess) {
                GGML_ABORT("HIP<hipLaunchKernelGGL> failed: %s\n", hipGetErrorString(res));
            }
            // compute
            hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matmul<DEBUT|FIN,M2,N2,K2,M1,N1,K1>),
                    dim3(nb_block<N0*N1*N2>(N),M/(M0*M1*M2),1), dim3(32, M2, N2),
                    0, 0,
                    A,B_cache.m_data,C,(float32_t*)nullptr, M,N,K);
            res = hipGetLastError();
            if (res != hipSuccess) {
                GGML_ABORT("HIP<hipLaunchKernelGGL> failed: %s\n", hipGetErrorString(res));
            }
        } else {
            for (std::size_t k3=0; k3<K; k3+=_K2) {
                // repack B
                hipLaunchKernelGGL(HIP_KERNEL_NAME(pack<1,_K2/KB>),
                        dim3(nb_block<N0>(N),1,1), dim3(32, 1, 1),
                        0, 0,
                        &B[k3],B_cache.m_data, N,K);
                auto res = hipGetLastError();
                if (res != hipSuccess) {
                    GGML_ABORT("HIP<hipLaunchKernelGGL> failed: %s", hipGetErrorString(res));
                }
                // compute
                if (k3 == 0) {
                    // le debut
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matmul<DEBUT,M2,N2,K2,M1,N1,K1>),
                            dim3(nb_block<N0*N1*N2>(N),M/(M0*M1*M2),1), dim3(32, M2, N2),
                            0, 0,
                            &A[k3*M],B_cache.m_data,C,C_cache.m_data, M,N,K);
                } else if (k3 < K-_K2) {
                    // les milieux
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matmul<MILIEUX,M2,N2,K2,M1,N1,K1>),
                            dim3(nb_block<N0*N1*N2>(N),M/(M0*M1*M2),1), dim3(32, M2, N2),
                            0, 0,
                            &A[k3*M],B_cache.m_data,C,C_cache.m_data, M,N,K);
                } else {
                    // la fin
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matmul<FIN,M2,N2,K2,M1,N1,K1>),
                            dim3(nb_block<N0*N1*N2>(N),M/(M0*M1*M2),1), dim3(32, M2, N2),
                            0, 0,
                            &A[k3*M],B_cache.m_data,C,C_cache.m_data, M,N,K);
                }
                res = hipGetLastError();
                if (res != hipSuccess) {
                    GGML_ABORT("HIP<hipLaunchKernelGGL> failed: %s\n", hipGetErrorString(res));
                }
            }
        }
        HIP_CHECK_ERROR(hipDeviceSynchronize());
    }

    //----------------------------------
    // - gemv:
    template<std::size_t _M1, std::size_t _K1, std::size_t _M2, std::size_t _K2>
    void sgemv_wmma(const bfloat16_t* A, const float32_t* B, float32_t* C, std::size_t M, std::size_t K) {
        constexpr int M0 = 16;
        constexpr int KV = 2;
        constexpr int KB = 16;
        constexpr int K0 = KB/KV;

        constexpr int M1 = _M1;
        constexpr int K1 = _K1;

        constexpr int M2 = _M2;
        constexpr int K2 = _K2/(K1*KB);

        GGML_ASSERT((K%_K2) == 0);
        GGML_ASSERT((M%(M0*M1*M2)) == 0);

        if (K<=_K2) {
            // compute
            static_assert((DEBUT|FIN) == 3);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matvec<(DEBUT|FIN),M2,K2,M1,K1>),
                    dim3(M/(M0*M1*M2)), dim3(32, M2),
                    0, 0,
                    A,B,C, M,K);
            auto res = hipGetLastError();
            if (res != hipSuccess) {
                GGML_ABORT("HIP<hipLaunchKernelGGL> failed: %s\n", hipGetErrorString(res));
            }
        } else {
            for (std::size_t k3=0; k3<K; k3+=_K2) {
                // compute
                if (k3 == 0) {
                    // le debut
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matvec<DEBUT,M2,K2,M1,K1>),
                            dim3(M/(M0*M1*M2)), dim3(32, M2),
                            0, 0,
                            &A[k3*M],&B[k3],C, M,K);
                } else if (k3 < K-_K2) {
                    // les milieux
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matvec<MILIEUX,M2,K2,M1,K1>),
                            dim3(M/(M0*M1*M2)), dim3(32, M2),
                            0, 0,
                            &A[k3*M],&B[k3],C, M,K);
                } else {
                    // la fin
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matvec<FIN,M2,K2,M1,K1>),
                            dim3(M/(M0*M1*M2)), dim3(32, M2),
                            0, 0,
                            &A[k3*M],&B[k3],C, M,K);
                }
                auto res = hipGetLastError();
                if (res != hipSuccess) {
                    GGML_ABORT("HIP<hipLaunchKernelGGL> failed: %s\n", hipGetErrorString(res));
                }
            }
        }
        HIP_CHECK_ERROR(hipDeviceSynchronize());
    }
}

//##############################################################################################
// les methodes a instancier pour l'op mulmat de l' IGPU
namespace ggml::backend::igpu::op_mul_mat {

    template<typename TA, typename TB, typename TC> constexpr bool supported_op() { return false; }
    template<> constexpr bool supported_op<bfloat16_t, float32_t, float32_t>() { return true; }

    template<typename TIN, typename TOUT> constexpr bool supported_repack() { return false; }
    template<> constexpr bool supported_repack<bfloat16_t, bfloat16_t>() { return true; }

    // la config :
    // taille de repacking:
    static constexpr std::size_t BLOC_M0 = 16;
    // static constexpr std::size_t BLOC_N0 = 16;
    static constexpr std::size_t BLOC_K0 = 8;
    static constexpr std::size_t BLOC_KV = 2;
    static constexpr std::size_t BLOC_K1 = 1024; // @ voir si il y en a des diferent!

    //std::size_t K_MAX = 1;
    //std::size_t M_MAX = 1;
    //std::size_t N_MAX = 512; // taille optimale ?

	/*
    bool init_caches() {
        // @ voir...
        //if (B_cache.ensure_size(BLOC_K1 * 1024)) { // @ optimiser la taille de N < 768 * 16
        //    IGPU_TRACE("B_cache[" << BLOC_K1 << ", " << 1024 <<"]");
        //}
        return true;
    }
    */

    inline bool _supported(const ggml_tensor& A, const ggml_tensor& B, const ggml_tensor& C) {
        // limite de tailles @ voir si on accepte plus.
        if (A.ne[0] % BLOC_K1 != 0) return false;  // 1B en a a 2046
        if (A.ne[1] %     128 != 0) return false;  // 1B en a a  512

        // qq limites...
        if (A.ne[0]*A.ne[1] >= 0x80000000) {
            IGPU_TRACE( C.name << "(" << A.name << ") K*M trop grand: " << A.ne[0] << ", " << A.ne[1]);
            return false;
        }
        if ((A.ne[0] % BLOC_K1)  != 0) {
            IGPU_TRACE( C.name << ": K non supporte : " << BLOC_K1 << "/" << A.ne[1]);
            return false;
        }
        if ((A.ne[1] % (4*2*16)) != 0) {
            IGPU_TRACE( C.name << ": M non supporte : " << 4*2*16 << "/" << A.ne[2]);
            return false;
        }
        // memoriser les tailles max de M et K pour allocation des caches => voir comment / quand les utiliser
        // => il faut prandre les tailles qui sont utils voir ensure_size()
        /*
        if (K_MAX<A.ne[0]) {
            K_MAX=A.ne[0];
            IGPU_TRACE("K_MAX: " << K_MAX);
        }
        if (M_MAX<A.ne[1]) {
            M_MAX=A.ne[1];
            IGPU_TRACE("M_MAX: " << M_MAX);
        }
        */
        // ordre:
        //  - supports_op
        //  - init_tensor
        //  - set_tensor
        // GGML_LOG_INFO("ggml-igpu: MATMUL(%s): supported!\n", A->name);
        return true;
    }

    inline bool _repack(const bfloat16_t* ref, std::size_t la, bfloat16_t* bloc, std::size_t M, std::size_t K) {
        // Ca sera important quand on fera le codage en fp8...
#       pragma omp parallel for num_threads(2) collapse(2) //  private(tmp)
        for (std::size_t k2=0; k2<K; k2+=BLOC_K1) {
            for (std::size_t i1=0; i1<M; i1+=BLOC_M0) {
                for (std::size_t k1=0; k1<BLOC_K1; k1+=16) {
                    bfloat16_t tmp[16][16];
                    for (std::size_t i0=0; i0<BLOC_M0; i0++) {
#                       pragma omp simd
                        for (std::size_t k0=0; k0<16; k0++) {
                            tmp[i0][k0] = ref[pos2D(la, M, k2+k1+k0, i1+i0)];
                        }
                    }
                    for (std::size_t k0=0; k0<16; k0++) {
#                       pragma omp simd
                        for (std::size_t i0=0; i0<BLOC_M0; i0++) {
                            bloc[posBloc2D<BLOC_K1,BLOC_M0,BLOC_KV,TYPE_BLOC::PERFECT>(K, M, k2+k1+k0, i1+i0)] = tmp[i0][k0];
                        }
                    }
                }
            }
        }
        return true;
    }

    // compute
    inline bool _compute(const bfloat16_t* A, const float32_t* B, float32_t* C,
            std::size_t M, std::size_t N, std::size_t K,
            std::size_t la, std::size_t lb, std::size_t lc
    ) {
        GGML_ASSERT(K % BLOC_K1 == 0);
//#define BENCH_COMPUTE
#ifdef BENCH_COMPUTE
        if (M%(2*8*BLOC_M0)==0) { // M=256
            //-sgemm_wmma<2,1,2,8,2,BLOC_K1>(A,B,C, M,N,K);
            //-sgemm_wmma<2,1,2,4,2,BLOC_K1>(A,B,C, M,N,K);
            //-sgemm_wmma<2,1,2,8,4,BLOC_K1>(A,B,C, M,N,K);
            // sgemm_wmma<2,1,2,4,4,BLOC_K1>(A,B,C, M,N,K);
            sgemm_wmma<2,2,2,4,4,BLOC_K1>(A,B,C, M,N,K);
            return true;
        } else {
            return false;
        }
#endif
        if (N <= 0) {
            return true;
        } else if (N==1) { // matvect[4096, 1, 4096]<2,1,1,8,1,1024>
            if (M%(2*8*BLOC_M0)==0) { // M=256
                sgemv_wmma<2,1,8,BLOC_K1>(A,B,C, M,K);
            } else if (M%(1*8*BLOC_M0)==0) { // M=128
                sgemv_wmma<2,1,4,BLOC_K1>(A,B,C, M,K);
            } else {
                // on va s'arreter la pour l'instant:
                return false;
            }
        } else if (N<=16) { // matmul[4096, 2:16, 4096]<1,1,2,8,1,1024>
            if (M%(1*8*BLOC_M0)==0) { // M=128
                sgemm_wmma<1,1,2,8,1,BLOC_K1>(A,B,C, M,N,K);
            } else {
                // on va s'arreter la pour l'instant:
                return false;
            }
        } else if (N<32) { // matmul[4096, 17, 4096]<1,2,2,8,1,1024>
            if (M%(1*8*BLOC_M0)==0) { // M=128
                sgemm_wmma<1,2,2,8,1,BLOC_K1>(A,B,C, M,N,K);
            } else {
                // on va s'arreter la pour l'instant:
                return false;
            }
            // quel autre cas? ... entre [32 et 128[
        } else if (N<=48) {
            if (M%(1*8*BLOC_M0)==0) { // M=128
                sgemm_wmma<2,1,2,8,4,BLOC_K1>(A,B,C, M,N,K);
            } else {
                // on va s'arreter la pour l'instant:
                return false;
            }
        } else if (N<=176) { // ???
            // matmul[4096, 32, 4096]<2,1,2,8,2,1024>
            // matmul[4096, 64, 4096]<2,1,2,4,4,1024>
            if (M%(1*8*BLOC_M0)==0) { // M=128
                sgemm_wmma<2,1,2,4,4,BLOC_K1>(A,B,C, M,N,K);
            } else {
                // on va s'arreter la pour l'instant:
                return false;
            }
        } else { // matmul[4096, 512, 4096]<2,2,2,4,4,1024>
            if (M%(2*4*BLOC_M0)==0) { // M=128
                sgemm_wmma<2,2,2,4,4,BLOC_K1>(A,B,C, M,N,K);
            } else {
                // on va s'arreter la pour l'instant:
                return false;
            }
        }
        //ggml::backend::igpu::sgemm_wmma<M1,N1,K1,M2,N2,K2>(a1,b1,c1,  M,N,K);

        return true;
    }


    //===========================================================================================================
    // les template a implementer.
    template<typename TA, typename TB, typename TC>
    bool supported(const ggml_tensor& A, const ggml_tensor& B, const ggml_tensor& C) {
        if constexpr (supported_op<TA,TB,TC>()) {
            return _supported(A,B,C);
        } else {
            return false;
        }
    }

    template<typename TIN, typename TOUT>
    bool repack(const TIN* A, std::size_t la, TOUT* bloc, std::size_t M, std::size_t K) {
        if constexpr (supported_repack<TIN,TOUT>()) {
            return _repack(A, la, bloc, M, K);
        } else {
            return false;
        }
    }

    template<typename TA, typename TB, typename TC>
    bool compute(const TA* A, const TB* B, TC* C,
            std::size_t M, std::size_t N, std::size_t K,
            std::size_t la, std::size_t lb, std::size_t lc
    ) {
        if constexpr (supported_op<TA,TB,TC>()) {
            return _compute(A,B,C, M,N,K, la,lb,lc);
        } else {
            return false;
        }
    }
}
