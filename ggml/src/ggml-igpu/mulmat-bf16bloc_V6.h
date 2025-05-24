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
    using float16x2_t   = _Float16 __attribute__((ext_vector_type(2)));
    using float16x16_t  = _Float16 __attribute__((ext_vector_type(16)));
    using float32x8_t   =  float __attribute__((ext_vector_type(8)));

    using hipint_t = int;

    //template<int I, typename T1, typename... T> struct select_type;
    template<int I, typename T1, typename... T>
    struct select_type {
        using type = select_type<I-1, T...>::type;
    };
    template<typename T1, typename... T>
    struct select_type<1,T1,T...> {
        using type = T1;
    };

    // le type a utilisé pour les calcules/conversion en fonction des types d'entrées...
    template<typename TA, typename TB, typename TC> struct Types {
        static_assert(false, "Types non supportés");
    };

    template<> struct Types<float32_t, float32_t, float32_t> {
        // voir a mettre des infos comme les tailles des Vecteurs à considerer... KV / K0,M0,M0
        static constexpr int KV = 2;
        static constexpr int K0 = 8; // 16/2 ...
        static constexpr int N0 = 16;
        static constexpr int M0 = 16;
        static constexpr bool CONV_A = true;
        static constexpr bool CONV_B = true;
        using ta = float32_t;
        using tb = float32_t;
        using tc = float32_t;
        using a_t     = bfloat16_t;    // ou float16_t ...
        using avect_t = bfloat16x2_t;  // ou float16x2_t ...
        using afrag_t = bfloat16x16_t; // ou float16x16_t ...
        using b_t     = a_t;
        using bvect_t = avect_t;
        using bfrag_t = afrag_t;
        using c_t     = float32_t;
        using cfrag_t = float32x8_t;
    };

    template<> struct Types<bfloat16_t, float32_t, float32_t> {
        static constexpr int KV = 2;
        static constexpr int K0 = 8; // 16/2 ...
        static constexpr int N0 = 16;
        static constexpr int M0 = 16;
        static constexpr bool CONV_A = false;
        static constexpr bool CONV_B = true;
        using ta = bfloat16_t;
        using tb = float32_t;
        using tc = float32_t;
        using a_t     = bfloat16_t;
        using avect_t = bfloat16x2_t;
        using afrag_t = bfloat16x16_t;
        using b_t     = a_t;
        using bvect_t = avect_t;
        using bfrag_t = afrag_t;
        using c_t     = float32_t;
        using cfrag_  = float32x8_t;
    };

    template<> struct Types<bfloat16_t, bfloat16_t, float32_t> {
        static constexpr int KV = 2;
        static constexpr int K0 = 8; // 16/2 ...
        static constexpr int N0 = 16;
        static constexpr int M0 = 16;
        static constexpr bool CONV_A = false;
        static constexpr bool CONV_B = false;
        using ta = bfloat16_t;
        using tb = bfloat16_t;
        using tc = float32_t;
        using a_t     = bfloat16_t;
        using avect_t = bfloat16x2_t;
        using afrag_t = bfloat16x16_t;
        using b_t     = a_t;
        using bvect_t = avect_t;
        using bfrag_t = afrag_t;
        using c_t     = float32_t;
        using cfrag_t = float32x8_t;
    };

    template<> struct Types<float16_t, float32_t, float32_t> {
        static constexpr int KV = 2;
        static constexpr int K0 = 8; // 16/2 ...
        static constexpr int N0 = 16;
        static constexpr int M0 = 16;
        static constexpr bool CONV_A = false;
        static constexpr bool CONV_B = true;
        using ta = float16_t;
        using tb = float32_t;
        using tc = float32_t;
        using a_t     = float16_t;
        using avect_t = float16x2_t;
        using afrag_t = float16x16_t;
        using b_t     = a_t;
        using bvect_t = avect_t;
        using bfrag_t = afrag_t;
        using c_t     = float32_t;
        using cfrag_t = float32x8_t;
    };

    template<> struct Types<float16_t, float16_t, float32_t> {
        static constexpr int KV = 2;
        static constexpr int K0 = 8; // 16/2 ...
        static constexpr int N0 = 16;
        static constexpr int M0 = 16;
        static constexpr bool CONV_A = false;
        static constexpr bool CONV_B = false;
        using ta = float16_t;
        using tb = float16_t;
        using tc = float32_t;
        using a_t     = float16_t;
        using avect_t = float16x2_t;
        using afrag_t = float16x16_t;
        using b_t     = a_t;
        using bvect_t = avect_t;
        using bfrag_t = afrag_t;
        using c_t     = float32_t;
        using cfrag_t = float32x8_t;
    };

    template<typename BASE>
    struct type_t { };

    template<> struct type_t<float32_t> {
        using t = float32_t;
        using x8 = float __attribute__((ext_vector_type(8)));
    };
    template<> struct type_t<bfloat16_t> {
        using t = bfloat16_t;
        using x2  = bfloat16x2_t;
        using x16 = bfloat16x16_t;
    };
    template<> struct type_t<float16_t> {
        using t = float16_t;
        using x2  = float16x2_t;
        using x16 = float16x16_t;
    };

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

    template<typename T_OUT, typename T_IN> __device__ T_OUT conv2t(T_IN val);
    template<> __device__ inline bfloat16_t conv2t<bfloat16_t, float32_t >(float32_t  val) { return val; }
    template<> __device__ inline bfloat16_t conv2t<bfloat16_t, bfloat16_t>(bfloat16_t val) { return val; }
    template<> __device__ inline float16_t  conv2t<float16_t,  float32_t >(float32_t  val) { return val; }
    template<> __device__ inline float16_t  conv2t<float16_t,  float16_t >(float16_t  val) { return val; }

    template<> __device__ inline bfloat16x2_t conv2t<bfloat16x2_t, bfloat16x2_t>(bfloat16x2_t val) { return val; }
    template<> __device__ inline float16x2_t  conv2t<float16x2_t,  float16x2_t >(float16x2_t  val) { return val; }

    // les wmma
    __device__ type_t<float32_t>::x8 __builtin_amdgcn_wmma(
            const type_t<bfloat16_t>::x16 a_frag, const type_t<bfloat16_t>::x16 b_frag, type_t<float32_t>::x8 c_frag)
    {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, c_frag);
    }
    __device__ type_t<float32_t>::x8 __builtin_amdgcn_wmma(
            const type_t<float16_t>::x16 a_frag, const type_t<float16_t>::x16 b_frag, type_t<float32_t>::x8 c_frag)
    {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);
    }

    // les dot2
    __device__ type_t<float32_t>::t __builtin_amdgcn_fdot(
            const type_t<bfloat16_t>::x2 a_frag, const type_t<bfloat16_t>::x2 b_frag, type_t<float32_t>::t c_frag)
    {
        return __builtin_amdgcn_fdot2_f32_bf16(a_frag, b_frag, c_frag, false);
    }
    __device__ type_t<float32_t>::t __builtin_amdgcn_fdot(
            const type_t<float16_t>::x2 a_frag, const type_t<float16_t>::x2 b_frag, type_t<float32_t>::t c_frag)
    {
        return __builtin_amdgcn_fdot2(a_frag, b_frag, c_frag, false);
    }

    //=====================================================================================
    // les kernels:
    //----------------------------------
    // - gemm:
    //   - reformat de B depuis le format ggml avec conversion
    template<hipint_t N1, hipint_t K1, hipint_t N0=16, hipint_t K0=8, hipint_t KV=2,
            typename TB_IN, typename TB_OUT>
    __global__ void __launch_bounds__(32) pack(
            const TB_IN* __restrict__ b, TB_OUT* __restrict__ b_cache,
            hipint_t N, hipint_t K
    ) {
        using conv_t  = type_t<TB_OUT>::t;
        using conv2_t = type_t<TB_OUT>::x2;

        const hipint_t tid = threadIdx.x;
        constexpr hipint_t NB_THREAD = 32; // blockDim.x

        const hipint_t j2 = blockIdx.x;
        const hipint_t J0 = j2*N0*N1; // + [0..N0*N1]

        __shared__ conv2_t B_trans[K1*K0][N0+1];
        conv2_t* __restrict__ B_cache = (conv2_t* __restrict__)b_cache;

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
                    B_trans[k0][j0][kv] = conv2t<conv_t>(b[k0*KV+kv + j*K]);
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
                    B_cache[j0 + k0*N0 + j1*N0*K0*K1 + j2*N0*K0*K1*N1] = B_trans[k0][j0];
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
    typename TA, typename TB, typename TC>
    __global__ void __launch_bounds__(16*2*M2*N2) wmma_matmul(
            const TA* __restrict__ a, const TB* __restrict__ b, TC* __restrict__ c,
            TC* __restrict__ c_cache,
            hipint_t M, hipint_t N, hipint_t K
    )
    {
        using t = Types<TA, TB, TC>;
        using a_comp_t = t::a_t;
        using b_comp_t = t::b_t;
        using c_comp_t = t::c_t;
        using a_vect_t = t::avect_t;
        using b_vect_t = t::bvect_t;
        using a_frag_t = t::afrag_t;
        using b_frag_t = t::bfrag_t;
        using c_frag_t = t::cfrag_t;

        // only possible values!
        static_assert(M0==t::M0, "use of __builtin_amdgcn_wmma");
        static_assert(N0==t::N0, "use of __builtin_amdgcn_wmma");
        static_assert(KV==t::KV, "use of __builtin_amdgcn_wmma");
        static_assert(K0==t::K0, "use of __builtin_amdgcn_wmma");
        static_assert((KV*K0)==16, "use of __builtin_amdgcn_wmma");
        
        static_assert(t::CONV_A==false, "pas encore de conversion possible pour A");
        static_assert(t::CONV_B==false, "pas de conversion possible pour B => doit etre faite par le pack!");
        
        using A_t = const a_vect_t[M2][M1][K2*K1*K0][M0];
        using B_t = const b_vect_t[N2][N1][K2*K1*K0][N0];

        //                                [ NB_THREAD   ]
        using C_t = c_comp_t[N1][M1][M0/2][N2][M2][2][N0];

        const hipint_t tid = threadIdx.x + 32 * threadIdx.y + 32*M2*threadIdx.z;
        constexpr hipint_t NB_THREAD = 32*M2*N2;
        const hipint_t i2 = threadIdx.y;
        const hipint_t j2 = threadIdx.z;

        static_assert( (N0*K0*N1*N2*K1)%NB_THREAD == 0);
        static_assert( (M0*K0*M1*M2*K1)%NB_THREAD == 0);
        static_assert( (M0*M2*N0*N2)%NB_THREAD == 0);

        // Voir comment optimiser ca...
        // sans L3 pas facile... il y a que la L2 pour ca...
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
        using A_frag_t  = a_vect_t[K1][M2][M1][K0][M0];
        using B_frag_t  = b_vect_t[K1][N2][N1][K0][N0];
        using C_frag_t  = c_comp_t[N0][M2*M1*M0+1];

        // TODO: tester avec FlipFlop ...
        __shared__ char data[HIP_MAX((sizeof(A_frag_t) + sizeof(B_frag_t))*2, sizeof(C_frag_t))];
        A_frag_t*  A_frag  = reinterpret_cast<A_frag_t*> (&data[0]);
        B_frag_t*  B_frag  = reinterpret_cast<B_frag_t*> (&data[sizeof(A_frag_t)*2]);
        C_frag_t&  C_frag  = *reinterpret_cast<C_frag_t*> (&data[0]);

        // chargement de C (sauf debut)
        c_frag_t c_frag[N1][M1] = {};  // [i0:i0+16] => [j0:j0+16 & i1:i1+M1 / thread]

        // chargement de A
        for (hipint_t id = tid; id<M0*K0*M1*M2*K1; id+=NB_THREAD) {
            const hipint_t i0 =  id%M0;
            const hipint_t k0 = (id/M0)%K0;
            const hipint_t k1 = (id/(M0*K0))%K1;
            const hipint_t i1 = (id/(M0*K0*K1))%M1;
            const hipint_t i2 =  id/(M0*K0*K1*M1);
            // A => 
            A_frag[0][k1][i2][i1][k0][i0] = A[i2][i1][k0+k1*K0][i0];
        }
        // chargement de B
        for (hipint_t id = tid; id<N0*K0*K1*N1*N2; id+=NB_THREAD) {
            const hipint_t j0 =  id%N0;
            const hipint_t k0 = (id/N0)%K0;
            const hipint_t k1 = (id/(N0*K0))%K1;
            const hipint_t j1 = (id/(N0*K0*K1))%N1;
            const hipint_t j2 =  id/(N0*K0*K1*N1);
            if(j2*N0*N1+ j3*N0*N1*N2 < N) {
                B_frag[0][k1][j2][j1][k0][j0] = B[j2][j1][k0+k1*K0][j0];
            }
        }
        __syncthreads(); // OK il faut attendre que tout soit reformaté.

        // calcul
        for (hipint_t k2=0; k2<K2; ++k2) {
            const hipint_t use  = (k2  )%2;
            const hipint_t load = (k2+1)%2;
            if (k2+1<K2) {
                // chargement de A
                for (hipint_t id = tid; id<M0*K0*M1*M2*K1; id+=NB_THREAD) {
                    const hipint_t i0 =  id%M0;
                    const hipint_t k0 = (id/M0)%K0;
                    const hipint_t k1 = (id/(M0*K0))%K1;
                    const hipint_t i1 = (id/(M0*K0*K1))%M1;
                    const hipint_t i2 =  id/(M0*K0*K1*M1);
                    // A => 
                    A_frag[load][k1][i2][i1][k0][i0] = A[i2][i1][k0+k1*K0+(k2+1)*K0*K1][i0];
                }
            }

            if (J0<N) {
                // calcul:
                a_comp_t a_frag[K0*KV];             // [M1xM0/thread]
                using a_frag_in_t  = a_vect_t[K0];  // pour le load
                using a_frag_out_t = a_frag_t;      // pour le calcul
                auto& a_frag_in  = reinterpret_cast<a_frag_in_t &>(a_frag);
                auto& a_frag_out = reinterpret_cast<a_frag_out_t&>(a_frag);

                a_comp_t b_frag[N1][K0*KV];           // [N0/thread]  => copie sur M1
                using b_frag_in_t  = b_vect_t[N1][K0];
                using b_frag_out_t = b_frag_t[N1];
                auto& b_frag_in  = reinterpret_cast<b_frag_in_t &>(b_frag);
                auto& b_frag_out = reinterpret_cast<b_frag_out_t&>(b_frag);

                for (hipint_t k1=0; k1<K1; ++k1) {
                    // chargement de b_frag (le meme pour chaque i1)
                    for (hipint_t j1=0; j1<N1; ++j1) {
                        if ((J0+j1*N0)<N) {
#                           pragma unrool K0
                            for (int k0=0; k0<K0; ++k0) {
                                b_frag_in[j1][k0] = B_frag[use][k1][j2][j1][k0][j0];
                            }
                        }
                    }

                    for (hipint_t i1=0; i1<M1; ++i1) {
                        // chargement de a_frag
#                       pragma unrool K0
                        for (int k0=0; k0<K0; ++k0) {
                            a_frag_in[k0] = A_frag[use][k1][i2][i1][k0][i0];
                        }
                        for (hipint_t j1=0; j1<N1; ++j1) {
                            c_frag[j1][i1] = __builtin_amdgcn_wmma(a_frag_out, b_frag_out[j1], c_frag[j1][i1]);
                        }
                    }
                }
            }

            if (k2+1<K2) {
                // chargement de B
                for (hipint_t id = tid; id<N0*K0*K1*N1*N2; id+=NB_THREAD) {
                    const hipint_t j0 =  id%N0;
                    const hipint_t k0 = (id/N0)%K0;
                    const hipint_t k1 = (id/(N0*K0))%K1;
                    const hipint_t j1 = (id/(N0*K0*K1))%N1;
                    const hipint_t j2 =  id/(N0*K0*K1*N1);
                    if(j2*N0*N1+ j3*N0*N1*N2 < N) {
                        B_frag[load][k1][j2][j1][k0][j0] = B[j2][j1][k0+k1*K0+(k2+1)*K0*K1][j0];
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
    hipint_t K2, hipint_t M1, hipint_t N1=1,
    hipint_t M0=16,hipint_t KV=2,
    typename TA, typename TB, typename TC>
    __global__ void __launch_bounds__(M0*M1) dot2_matmul(
            const TA* __restrict__ a, const TB* __restrict__ b, TC* __restrict__ c,
            hipint_t M, hipint_t K)
    {
        using t = Types<TA, TB, TC>;
        using a_comp_t = t::a_t;
        using b_comp_t = t::b_t;
        using a_frag_t = t::avect_t;
        using b_frag_t = t::bvect_t;
        using c_frag_t = t::c_t;
        
        // only possible/interesting values!
        static_assert(KV==t::KV, "use of __builtin_amdgcn_fdot2");
        static_assert(M0==t::M0, "use of __builtin_amdgcn_wmma");

        using A_t = select_type<t::CONV_A?1:2, const TA[M1][K2][M0][KV], const a_frag_t[M1][K2][M0]>::type;
        //using B_t = const TB[N1][K];

        constexpr hipint_t NB_THREAD = M0*M1;
        const hipint_t tid = threadIdx.x + M0*threadIdx.y;
        const hipint_t i0 = threadIdx.x;
        const hipint_t i1 = threadIdx.y;
        const hipint_t i3 = blockIdx.x;

        // pour les "calcul"
        A_t& __restrict__ A = reinterpret_cast<A_t*> (a)[i3];

        // les shareds:
        using B_frag_t = b_frag_t[N1][K2];
        using C_frag_t = c_frag_t[M1][M0];

        __shared__ B_frag_t B_frag;
        __shared__ C_frag_t C_frag;

        // le calcul de C sur K0*K1*K2
        c_frag_t c_frag[N1] = {};  // M0*M1 thread...

        // chargement de B => shared
        for (hipint_t j1=0; j1<N1; ++j1) {
            for (hipint_t k2 = tid; k2<K2; k2+=NB_THREAD) {
#               pragma unrool KV
                for (hipint_t kv=0; kv<KV; ++kv) {
                    B_frag[j1][k2][kv] = conv2t<b_comp_t>(b[k2*2+kv + j1*K]);
                }
            }
        }

        __syncthreads(); // OK il faut attendre que tout soit chargé.

        // calcul
        for (hipint_t k2=0; k2<K2; ++k2) {
            a_frag_t a_frag = conv2t<a_frag_t>(A[i1][k2][i0]);
            for (hipint_t j1=0; j1<N1; ++j1) {
                b_frag_t b_frag = B_frag[j1][k2];
                c_frag[j1] = __builtin_amdgcn_fdot(a_frag, b_frag, c_frag[j1]);
            }
        }

        // ecriture de C
        for (hipint_t j1=0; j1<N1; ++j1) {
            if constexpr(TYPE ==  DEBUT     ) { c[i0+i1*M0+i3*M0*M1 + j1*M]  = c_frag[j1]; }
            if constexpr(TYPE == (DEBUT|FIN)) { c[i0+i1*M0+i3*M0*M1 + j1*M]  = c_frag[j1]; }
            if constexpr(TYPE ==    MILIEUX ) { c[i0+i1*M0+i3*M0*M1 + j1*M] += c_frag[j1]; }
            if constexpr(TYPE ==        FIN ) { c[i0+i1*M0+i3*M0*M1 + j1*M] += c_frag[j1]; }
        }
    }

    //=====================================================================================
    template<std::size_t N> constexpr std::size_t block_size(std::size_t size) { return (((size-1)/N) + 1) * N; }
    template<std::size_t N> constexpr std::size_t nb_block  (std::size_t size) { return (((size-1)/N) + 1); }

    class hip_cache {
        void clear() {
            if (m_size > 0) {
                ggml::hip::deallocateDevice((std::uint8_t*)m_data);
                m_data = nullptr;
            }
            m_size = 0;
        }
    public:
        hip_cache() {}
        ~hip_cache() { clear(); }
        template<typename T>
        void min_size(std::size_t size) {
            size *= sizeof(T);
            // permet de definir une taille min a allouer sans l'allouer encore
            if (size>m_sizeMin) m_sizeMin = size;
        }
        template<typename T>
        bool ensure_size(std::size_t size) {
            // on alloue ce qu'il faut (max(size, m_sizeMin));
            min_size<T>(size);
            if (m_sizeMin > m_size) {
                clear();
                m_size = m_sizeMin;
                m_data = ggml::hip::allocateDevice<std::uint8_t*>(m_size);
                return true;
            }
            return false;
        }
        template<typename T>
        T* data() {
            return reinterpret_cast<T*>(m_data);
        }
    private:
        void* m_data = nullptr; // not realy a T* ... it is a ref on device memory
        std::size_t m_size = 0;
        std::size_t m_sizeMin = 0;
    };

    template<typename T>
    class hip_cache_t {
    public:
        hip_cache_t(hip_cache& cache): m_cache(cache) { }
        void min_size(std::size_t size) {
            // std::cout << "####>>> hip_cache_t min_size: " << size << std::endl;
            m_cache.min_size<T>(size);
        }
        bool ensure_size(std::size_t size) {
            //std::cout << "####>>> hip_cache_t ensure_size: " << size << std::endl;
            return m_cache.ensure_size<T>(size);
        }
        T* data() {
            return m_cache.data<T>();
        }
    private:
        hip_cache& m_cache;
    };

    // @ initialiser a l'init du backend en fonction de la taille des poids?
    //  TODO: voir comment mutualiser entre les types la 1 static par fichier ?
    static hip_cache s_B_cache;
    static hip_cache s_C_cache;

    //=====================================================================================
    // les fonctions hotes
    //----------------------------------
    // - gemm:
    template<std::size_t _M1, std::size_t _N1, std::size_t _K1, std::size_t _M2, std::size_t _N2, std::size_t _K2,
             typename TA, typename TB, typename TC>
    void sgemm_wmma(const TA* A, const TB* B, TC* C, std::size_t M, std::size_t N, std::size_t K) {
        using t = Types<TA, TB, TC>;
        using a_comp_t = t::a_t;
        using b_comp_t = t::b_t;
        using c_comp_t = t::c_t;
        
        hip_cache_t<b_comp_t> B_cache(s_B_cache);
        hip_cache_t<c_comp_t> C_cache(s_C_cache);

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
        if (B_cache.ensure_size(_K2 * block_size<N0*N1*N2>(N))) {
            IGPU_TRACE("B_cache[" << _K2 << ", " << block_size<N0*N1*N2>(N) <<"]");
            //std::cout << "#> ggml-igpu: " << "B_cache[" << _K2 << ", " << block_size<N0*N1*N2>(N) <<"]" << std::endl;
        }
        if (C_cache.ensure_size(block_size<N0*N1*N2>(N) * block_size<M0*M1*M2>(M))) {
            IGPU_TRACE("C_cache[" << block_size<M0*M1*M2>(M) << ", " << block_size<N0*N1*N2>(N) <<"]");
            //std::cout << "#> ggml-igpu: " << "C_cache[" << block_size<M0*M1*M2>(M) << ", " << block_size<N0*N1*N2>(N) <<"]" << std::endl;
        }

        if (K<=_K2) {
            // repack B
            hipLaunchKernelGGL(HIP_KERNEL_NAME(pack<N1*N2,_K2/KB>),
                    dim3(nb_block<N0*N1*N2>(N),1,1), dim3(32, 1, 1),
                    0, 0,
                    B,B_cache.data(), N,K);
            auto res = hipGetLastError();
            if (res != hipSuccess) {
                GGML_ABORT("HIP<hipLaunchKernelGGL> failed: %s\n", hipGetErrorString(res));
            }
            // compute
            hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matmul<DEBUT|FIN,M2,N2,K2,M1,N1,K1>),
                    dim3(nb_block<N0*N1*N2>(N),M/(M0*M1*M2),1), dim3(32, M2, N2),
                    0, 0,
                    A,B_cache.data(),C,(float32_t*)nullptr, M,N,K);
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
                        &B[k3],B_cache.data(), N,K);
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
                            &A[k3*M],B_cache.data(),C,C_cache.data(), M,N,K);
                } else if (k3 < K-_K2) {
                    // les milieux
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matmul<MILIEUX,M2,N2,K2,M1,N1,K1>),
                            dim3(nb_block<N0*N1*N2>(N),M/(M0*M1*M2),1), dim3(32, M2, N2),
                            0, 0,
                            &A[k3*M],B_cache.data(),C,C_cache.data(), M,N,K);
                } else {
                    // la fin
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matmul<FIN,M2,N2,K2,M1,N1,K1>),
                            dim3(nb_block<N0*N1*N2>(N),M/(M0*M1*M2),1), dim3(32, M2, N2),
                            0, 0,
                            &A[k3*M],B_cache.data(),C,C_cache.data(), M,N,K);
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
    // - gemm for small N:
    template<std::size_t _M1, std::size_t _N1, std::size_t _K2, typename TA, typename TB, typename TC>
    void sgemm_dot2(const TA* A, const TB* B, TC* C, std::size_t M, std::size_t N, std::size_t K) {
        constexpr int KV = 2;
        static_assert( _K2%KV == 0);

        constexpr int M0 = 16;
        constexpr int M1 = _M1;
        constexpr int N1 = _N1;
        constexpr int K2 = _K2/KV;

        GGML_ASSERT(N == N1);
        GGML_ASSERT((K%_K2) == 0);
        GGML_ASSERT((M%(M0*M1)) == 0);

        if (K<=_K2) {
            // compute
            static_assert((DEBUT|FIN) == 3);
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dot2_matmul<(DEBUT|FIN),K2,M1,N1>),
                    dim3(M/(M0*M1)), dim3(M0, M1),
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
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(dot2_matmul<DEBUT,K2,M1,N1>),
                            dim3(M/(M0*M1)), dim3(M0, M1),
                            0, 0,
                            &A[k3*M],&B[k3],C, M,K);
                } else if (k3 < K-_K2) {
                    // les milieux
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(dot2_matmul<MILIEUX,K2,M1,N1>),
                            dim3(M/(M0*M1)), dim3(M0, M1),
                            0, 0,
                            &A[k3*M],&B[k3],C, M,K);
                } else {
                    // la fin
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(dot2_matmul<FIN,K2,M1,N1>),
                            dim3(M/(M0*M1)), dim3(M0, M1),
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

// quel format pour faire ca simplement?
#define SGEMM_8(_N) \
     if (M%( 8*BLOC_M0)==0) { sgemm_dot2< 8,_N,BLOC_K1>(A,B,C, M,N,K); return true;}\
     return false

#define SGEMM_16(_N) \
     if (M%(16*BLOC_M0)==0) { sgemm_dot2<16,_N,BLOC_K1>(A,B,C, M,N,K); return true;}\
     if (M%( 8*BLOC_M0)==0) { sgemm_dot2< 8,_N,BLOC_K1>(A,B,C, M,N,K); return true;}\
     return false

#define SGEMM_32(_N) \
     if (M%(32*BLOC_M0)==0) { sgemm_dot2<32,_N,BLOC_K1>(A,B,C, M,N,K); return true;}\
     if (M%(16*BLOC_M0)==0) { sgemm_dot2<16,_N,BLOC_K1>(A,B,C, M,N,K); return true;}\
     if (M%( 8*BLOC_M0)==0) { sgemm_dot2< 8,_N,BLOC_K1>(A,B,C, M,N,K); return true;}\
     return false

//##############################################################################################
// les methodes a instancier pour l'op mulmat de l' IGPU
namespace ggml::backend::igpu::op_mul_mat {

    template<typename TA, typename TB, typename TC> constexpr bool supported_op() { return false; }
    template<> constexpr bool supported_op<bfloat16_t,  float32_t, float32_t>() { return true; }
    template<> constexpr bool supported_op< float16_t,  float32_t, float32_t>() { return true; }
    //template<> constexpr bool supported_op<bfloat16_t, bfloat16_t, float32_t>() { return true; }
    //template<> constexpr bool supported_op< float16_t,  float16_t, float32_t>() { return true; }

    template<typename TIN, typename TOUT> constexpr bool supported_repack() { return false; }
    template<> constexpr bool supported_repack<bfloat16_t, bfloat16_t>() { return true; }
    template<> constexpr bool supported_repack< float16_t,  float16_t>() { return true; }

    // la config :
    // taille de repacking:
    static constexpr std::size_t BLOC_M0 = 16;
    // static constexpr std::size_t BLOC_N0 = 16;
    static constexpr std::size_t BLOC_K0 = 8;
    static constexpr std::size_t BLOC_KV = 2;
    static constexpr std::size_t BLOC_K1 = 1024; // @ voir si il y en a des diferent!
    //static constexpr std::size_t BLOC_K1 =  256;  => pas bon!
    //static constexpr std::size_t BLOC_K1 =  512; => mieux mais pas le meilleur
    //static constexpr std::size_t BLOC_K1 = 2048;  // trop pour les dot2
    
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

    //===========================================================================================================
    // qq element global:
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
        //if (K_MAX<A.ne[0]) { K_MAX=A.ne[0]; IGPU_TRACE("K_MAX: " << K_MAX); }
        //if (M_MAX<A.ne[1]) { M_MAX=A.ne[1]; IGPU_TRACE("M_MAX: " << M_MAX); }
        // ordre:
        //  - supports_op
        //  - init_tensor
        //  - set_tensor
        // GGML_LOG_INFO("ggml-igpu: MATMUL(%s): supported!\n", A->name);

        return true;
    }
    // juste reformatage des types simples suivant M0,K1,KV ...
    //  TODO cas avec conversions ...
    template<typename T>
    bool _repack_simple(const T* ref, std::size_t la, T* bloc, std::size_t M, std::size_t K) {
        // Ca sera important quand on fera le codage en fp8...
#       pragma omp parallel for num_threads(2) collapse(2) //  private(tmp)
        for (std::size_t k2=0; k2<K; k2+=BLOC_K1) {
            for (std::size_t i1=0; i1<M; i1+=BLOC_M0) {
                for (std::size_t k1=0; k1<BLOC_K1; k1+=16) {
                    T tmp[16][16];
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
    // reste a savoir si c'est les meme optim suivant les quantisations
    //  - bf16 & fp16 => oui
    //  - fp32 ?
    //  - fp8 ?
    template<typename TA, typename TB, typename TC>
    bool _compute(const TA* A, const TB* B, TC* C,
            std::size_t M, std::size_t N, std::size_t K,
            std::size_t la, std::size_t lb, std::size_t lc
    ) {
        // bug dans supported si c'est pas le cas.
        GGML_ASSERT(K % BLOC_K1 == 0);
        GGML_ASSERT(M % 128 == 0);
//#define BENCH_COMPUTE
#ifdef BENCH_COMPUTE
        if (M%(2*8*BLOC_M0)==0) { // M=256
            // sgemm_wmma<2,1,2,8,2,BLOC_K1>(A,B,C, M,N,K);
            // sgemm_wmma<2,1,2,4,2,BLOC_K1>(A,B,C, M,N,K);
            // sgemm_wmma<2,1,2,8,4,BLOC_K1>(A,B,C, M,N,K);
            // sgemm_wmma<2,1,2,4,4,BLOC_K1>(A,B,C, M,N,K);
            sgemm_wmma<2,2,2,4,4,BLOC_K1>(A,B,C, M,N,K);
            return true;
        } else {
            return false;
        }
#endif

        // ggml::backend::igpu::sgemm_wmma<M1,N1,K1,M2,N2,K2>(a1,b1,c1,  M,N,K);
        // ggml::backend::igpu::sgemv_wmma<M1,N1,K2>(a1,b1,c1, M,N,K);
        if (N <= 0) { return true; } // bizard ca lui arrive de passer par la.
        // matvect[4096, 1, 4096]<2,8,1,1,1,1,1024>
        if (N == 1) { SGEMM_8( 1); }
        if (N == 2) { SGEMM_8( 2); }
        if (N == 3) { SGEMM_8( 3); }
        if (N == 4) { SGEMM_8( 4); }
        if (N == 5) { SGEMM_8( 5); }
        // matmul[4096, 6, 4096]<2,16,6,1,1,1,1024>
        if (N == 6) { SGEMM_16( 6); }
        if (N == 7) { SGEMM_16( 7); }
        if (N == 8) { SGEMM_16( 8); }
        if (N == 9) { SGEMM_16( 9); }
        if (N ==10) { SGEMM_16(10); }
        // matmul[4096, 11, 4096]<2,32,11,1,1,1,1024>
        if (N ==11) { SGEMM_32(11); }
        if (N ==12) { SGEMM_32(12); }
        if (N ==13) { SGEMM_32(13); }
        if (N ==14) { SGEMM_32(14); }
        if (N ==15) { SGEMM_32(15); }
        if (N ==16) { SGEMM_32(16); }
        if (N ==17) { SGEMM_32(17); }
        if (N ==18) { SGEMM_32(18); }
        if (N ==19) { SGEMM_32(19); }
        if (N ==20) { SGEMM_32(20); }
        if (N ==21) { SGEMM_32(21); }
        if (N ==22) { SGEMM_32(22); }
        if (N ==23) { SGEMM_32(23); }
        if (N ==24) { SGEMM_32(24); }
        if (N ==25) { SGEMM_32(25); }
        if (N ==26) { SGEMM_32(26); }
        if (N ==27) { SGEMM_32(27); }
        if (N ==28) { SGEMM_32(28); }
        if (N ==29) { SGEMM_32(29); }
        if (N ==30) { SGEMM_32(30); }
        if (N ==31) { SGEMM_32(31); }
        // quel autre cas... entre [32 et 128[
        if (N<32) {
            std::cout << "??? " << M << "," << N << "," << K << std::endl;
            return false;
        } else if (N<=48) {  // matmul[4096, 48, 4096]<2,1,2,8,4,1024>
            // M=256
            if (M%(2*8*BLOC_M0)==0) { sgemm_wmma<2,1,2,8,4,BLOC_K1>(A,B,C, M,N,K); return true; }
        } else if (N<=176) { // matmul[4096, 64, 4096]<2,1,2,4,4,1024>
            // M=128
            if (M%(2*4*BLOC_M0)==0) { sgemm_wmma<2,1,2,4,4,BLOC_K1>(A,B,C, M,N,K); return true; }
        } else { // matmul[4096, 512, 4096]<2,2,2,4,4,1024>
            // M=128
            if (M%(2*4*BLOC_M0)==0) { sgemm_wmma<2,2,2,4,4,BLOC_K1>(A,B,C, M,N,K); return true; }
        }
        // oups...
        std::cout << "??? " << M << "," << N << "," << K << std::endl;
        return false;
    }

    //===========================================================================================================
    // les template a implementer.
    template<typename TA, typename TB, typename TC>
    bool supported(const ggml_tensor& A, const ggml_tensor& B, const ggml_tensor& C) {
        if constexpr (supported_op<TA,TB,TC>()) {
            if (_supported(A,B,C)) {
                using t = Types<TA, TB, TC>;
                using a_comp_t = t::a_t;
                using b_comp_t = t::b_t;
                using c_comp_t = t::c_t;
                constexpr int M0 = 16;
                constexpr int N0 = 16;
                constexpr int M1 = 2;
                constexpr int N1 = 2;
                constexpr int M2 = 4;
                constexpr int N2 = 4;
                const std::size_t M = C.ne[0];
                const std::size_t N = C.ne[1];
                // gestion des caches:
                hip_cache_t<b_comp_t> B_cache(s_B_cache);
                hip_cache_t<c_comp_t> C_cache(s_C_cache);
                // les caches => voir a avoir un min pour N ~ 512?
                B_cache.min_size(BLOC_K1 * block_size<N0*N1*N2>(N));
                C_cache.min_size(block_size<N0*N1*N2>(N) * block_size<M0*M1*M2>(M));
                return true;
            }
            return false;
        } else {
            return false;
        }
    }

    template<typename TIN, typename TOUT>
    bool repack(const TIN* A, std::size_t la, TOUT* bloc, std::size_t M, std::size_t K) {
        if constexpr (supported_repack<TIN,TOUT>()) {
            return _repack_simple(A, la, bloc, M, K);
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
