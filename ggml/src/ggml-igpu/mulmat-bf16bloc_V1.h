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
    using bfloat16x16_t = __bf16 __attribute__((ext_vector_type(16)));
    using float32x8_t   =  float __attribute__((ext_vector_type(8)));

    template<std::size_t N>
    constexpr std::size_t block_size(std::size_t size) {
        return (((size-1)/N) + 1) * N;
    }
    template<std::size_t N>
    constexpr std::size_t nb_block(std::size_t size) {
        return (((size-1)/N) + 1);
    }

    using hipint_t = int;  // std::size_t => trop de SGPRs!

    template<typename... ARGS>
    constexpr __device__ hipint_t HIP_MAX(const hipint_t A, const hipint_t B, const ARGS... args) {
        if constexpr (sizeof...(args) == 0) {
            return A>B?A:B;
        } else {
            return HIP_MAX(A>B?A:B, args...);
        }
    }

    template<typename T> __device__ bfloat16_t conv2bf16(T val);
    template<> __device__ inline bfloat16_t conv2bf16(float32_t  val) { return val; }
    template<> __device__ inline bfloat16_t conv2bf16(bfloat16_t val) { return val; }

    // le kenel
    template<hipint_t M2, hipint_t M1, hipint_t N1, hipint_t K1,
    hipint_t M0=16, hipint_t N0=16, hipint_t K0=16,
    typename TA, typename TB, typename TC>
    __global__ void __launch_bounds__(16*2*M1) wmma_matmul(
            const TA* __restrict__ a, const TB* __restrict__ b, TC* __restrict__ c,
            hipint_t M, hipint_t N, hipint_t K,
            bfloat16_t* __restrict__ bc, hipint_t lb,
            float32_t*  __restrict__ cc, hipint_t lc
    )
    {
        // only possible values!
        static_assert(M0==16, "use of __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 => M0=16");
        static_assert(N0==16, "use of __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 => N0=16");
        static_assert(K0==16, "use of __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 => K0=16");

        // blockDim.x == 16 == M0/K0/N0 ...
        // blockDim.y = 2
        // blockDim.z == M1
        constexpr hipint_t NB_THREAD = M0*M1*2;
        const     hipint_t ID_THREAD = threadIdx.x + threadIdx.y*M0 + threadIdx.z*M0*2;
        // pour l'instant codé que si c'est vrai.
        static_assert(K1*K0>=NB_THREAD, "for B load simplicity: reduce M1 or increase K1");
        static_assert((K1*K0)%(NB_THREAD)==0, "for B load simplicity: reduce M1 or increase K1");

        const hipint_t K2 = K/(K1*K0);

        // A => A[K2=K/(K1*K0)][M/M0  == M3*M2*M1             ][K1 * K0][M0]
        //           k2         i4         i3           i2  i1  k1 | k0  i0
        //   => A[K2=K/(K1*K0)][M4][M3=M/(M4*M2*M1*M0)][M2][M1][K1]|[K0][M0]
        const     hipint_t i0 = threadIdx.x;
        const     hipint_t i1 = threadIdx.z;
        constexpr hipint_t strideA_M0 = 1;
        constexpr hipint_t strideA_K0 = M0;
        constexpr hipint_t strideA_K1 = M0*K0;
        constexpr hipint_t strideA_M1 = M0*K1*K0;
        const     hipint_t strideA_K2 = M*K1*K0;
        constexpr hipint_t strideA_M2 = M1*M0*K1*K0;
        const     hipint_t M3 = M/(gridDim.x*M2*M1*M0);
        // & = M%(gridDim.x*M2*M1*M0) == 0;
        constexpr hipint_t strideA_M3 = M2*M1*M0*K1*K0;
        const hipint_t M4 = gridDim.x;
        const hipint_t i4 = blockIdx.x;
        const hipint_t strideA_M4 = (M*K1*K0)/M4;    // pour passer de i4 a i4+1  // ici cas parfait? M%(M0*M1*M2) == 0

        // B => B[N=N2*N1*N0][K]
        //   => B_cache[M4]|[N1][K1*K0 ][N0]   / M4 pour eviter les colisions
        //                   i1  k1  k0  i0
        //   => B_cache[M4]|[N1][K1][K0][N0]
        const hipint_t j0 = threadIdx.x;
        const hipint_t N2 = ((N-1)/(N1*N0))+1; // un nombre de bloc suffisant
        const hipint_t j2 = blockIdx.y;

        // C => C[i,j] = C[j*lc + i] !
        //                  j2     i4       i3             i2  j1  i0b   i1    i0a   j0
        //   => C_cache[N/(N1*N0)][M4][M/(M0*M1*M2*M4]] / [M2][N1][M0/2][M1][M0/8=2][N0]
        //                      =M3
        const     hipint_t j0_c  = threadIdx.x;
        const     hipint_t i0a_c = threadIdx.y;
        const     hipint_t i1_c  = threadIdx.z;
        //const   integer_t j2    = blockIdx.y;
        constexpr hipint_t strideCb_N0  = 1;
        constexpr hipint_t strideCb_M0a = N0;
        constexpr hipint_t strideCb_M1  = N0*2;
        constexpr hipint_t strideCb_M0b = M1*N0*2;
        constexpr hipint_t strideCb_N1  = M1*N0*M0;
        constexpr hipint_t strideCb_M2  = N1*M1*N0*M0;
        constexpr hipint_t strideCb_M3  = M2*N1*M1*N0*M0;
        const     hipint_t strideCb_M4  = (M/M4)*N1*N0;
        const     hipint_t strideCb_N2  = M*N0*N1;

        using B_frag_t  = bfloat16_t[K1][N1][K0][N0];
        using B_trans_t = bfloat16_t[K1*K0][N0+2];
        using C_frag_t  = float32_t [N1*N0][M2*M1*M0+1];

        // le cache, il est reutilisé entre plusieurs etapes
        __shared__ char data[HIP_MAX(sizeof(B_frag_t), sizeof(B_trans_t), sizeof(C_frag_t))];
        B_frag_t&  B_frag  = * reinterpret_cast<B_frag_t*> (&data[0]);
        B_trans_t& B_trans = * reinterpret_cast<B_trans_t*>(&data[0]);
        C_frag_t&  C_out   = * reinterpret_cast<C_frag_t*> (&data[0]);

        // on avancera sur K...
        for (hipint_t k2=0; k2<K2; ++k2) {

            // re-formatage de B. (conversion + bloc)
            B_frag_t& __restrict__ BC = *reinterpret_cast<B_frag_t* __restrict__>(&bc[i4*N2*N1*N0*K1*K0 + j2*N1*N0*K1*K0]);
            for(hipint_t j1_b=0; j1_b<N1; ++j1_b) {
                const TB* __restrict__ B = &b[k2*K0*K1 + (j2*N0*N1 + j1_b*N0)*lb];
                __syncthreads(); // OK il faut attendre que tout soit calculé
                for(hipint_t j0_b=0; j0_b<N0; ++j0_b) {
                    if ((j0_b+j1_b*N0+j2*N0*N1) < N) {
                        for(hipint_t k1_b=0; k1_b<K1*K0; k1_b+=NB_THREAD) {
                            const hipint_t k0_b = ID_THREAD;
                            B_trans[k0_b+k1_b][j0_b] = conv2bf16(B[k0_b+k1_b + j0_b*lb]);
                        }
                    }
                }
                __syncthreads(); // OK il faut attendre que tout soit reformaté.
                for (hipint_t k1=0; k1<K1; ++k1) {
                    // il faut optimiser le chargement K0 => K0/M1 |  N0 => N0*M1
                    for (hipint_t x=0; x < K0 ; x+= (NB_THREAD/N0)) {  // (N0*K0)/NB_THREAD
                        const hipint_t k0 = threadIdx.y + threadIdx.z*2 + x;
                        BC[k1][j1_b][k0][j0] = B_trans[k0+k1*N0][j0];
                    }
                }
            }

            __syncthreads(); // OK il faut attendre que tout soit reformaté.

            // chargement de B_frag
            //  K0,N0,K1 par thread N1,K1,N0 en boucles... on peu faire 2 bf16 ou 1 fp32 ?
            for (hipint_t k1=0; k1<K1; ++k1) {
                for (hipint_t j1=0; j1<N1; ++j1) {
                    // il faut optimiser le chargement K0 => K0/M1 |  N0 => N0*M1
                    for (hipint_t x=0; x < K0 ; x+= (NB_THREAD/N0)) {  // (N0*K0)/NB_THREAD
                        const hipint_t k0 = threadIdx.y + threadIdx.z*2 + x;
                        B_frag[k1][j1][k0][j0] = BC[k1][j1][k0][j0];
                    }
                }
            }

            __syncthreads(); // OK il faut attendre que tout soit chargé.

            for (hipint_t i3=0; i3<M3; ++i3) {
                //                             j2     i4       i3          i2  j1  i0b      i1    i0a   j0
                // charger c_frag  <=    C[N/(N1*N0)][M4][M/(M0*M1*M2]] / [M2][N1][M0/2] | [M1][M0/8=2][N0]
                // les fragments a/b pour wmma / chaque thread "traite" M0/N0
                bfloat16x16_t a_frag;               // [k0..k0+K0[ / [M1xM0/thread]
                bfloat16x16_t b_frag[N1];           // [k0..k0+K0[ / [   N0/thread]  => copie sur M1 ?
                float32x8_t   c_frag[M2][N1] = {};  // [i0:i0+16] => [j0:j0+16 & i1:i1+M1 / thread]

                float32_t* __restrict__ C = &cc[j2*strideCb_N2 + i4*strideCb_M4 + i3*strideCb_M3 +
                                                j0_c*strideCb_N0 + i0a_c*strideCb_M0a + i1_c*strideCb_M1];

                // blocA suivant.
                const TA* __restrict__ A = &a[i4*strideA_M4 + i3*strideA_M3 + k2*strideA_K2 + i1*strideA_M1 + i0*strideA_M0]; // A[M2][M1][K1][K0][M0]

                if (k2>0) {
                    for (hipint_t i2=0; i2<M2; ++i2) {
                        for (hipint_t j1=0; j1<N1; ++j1) {
                            for (hipint_t i0b = 0; i0b<M0/2 ; ++i0b) {
                                c_frag[i2][j1][i0b] = C[i2*strideCb_M2 + j1*strideCb_N1 + i0b*strideCb_M0b];
                            }
                        }
                    }
                }

                for (hipint_t k1=0; k1<K1; ++k1) {
                    // chargement de b_frag (le meme pour chaque i1)
                    for (hipint_t j1=0; j1<N1; ++j1) {
#                       pragma unrool K0
                        for (int k0=0; k0<K0; k0++) {
                            b_frag[j1][k0] = B_frag[k1][j1][k0][j0];
                        }
                    }
                    for (hipint_t i2=0; i2<M2; ++i2) {
                        // chargement de a_frag
#                       pragma unrool K0
                        for (hipint_t k0=0; k0<K0; k0++) {
                            a_frag[k0] = A[i2*strideA_M2 + k1*strideA_K1 + k0*strideA_K0];
                        }
                        // calcul C=A*B:
#                       pragma unrool N1
                        for (hipint_t j1=0; j1<N1; ++j1) {
                            c_frag[i2][j1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag[j1], c_frag[i2][j1]);
                        }
                    }
                }

                //                            j2     i4       i3          i2  j1  i0b      i1    i0a   j0
                // sauver c_frag  <=    C[N/(N1*N0)][M4][M/(M0*M1*M2]] / [M2][N1][M0/2] | [M1][M0/8=2][N0]
                for (hipint_t i2=0; i2<M2; ++i2) {
                    for (hipint_t j1=0; j1<N1; ++j1) {
                        for (hipint_t i0b = 0; i0b<M0/2 ; ++i0b) {
                            C[i2*strideCb_M2 + j1*strideCb_N1 + i0b*strideCb_M0b] = c_frag[i2][j1][i0b];
                        }
                    }
                }
            }
        } // k2

        __syncthreads(); // OK il faut attendre que tout soit chargé.

        // ecriture de C:
        for (hipint_t i3=0; i3<M3; ++i3) {
            // recuperation d'un bloc de C depuis le cache
            float32_t* __restrict__ CC = &cc[j2*strideCb_N2 + i4*strideCb_M4 + i3*strideCb_M3 +
                                             j0_c*strideCb_N0 + i0a_c*strideCb_M0a + i1_c*strideCb_M1];
            for (hipint_t i2=0; i2<M2; ++i2) {
                for (hipint_t j1=0; j1<N1; ++j1) {
                    for (int i0b = 0; i0b<M0/2 ; ++i0b) {
                        const hipint_t pos1 = i2*strideCb_M2 + j1*strideCb_N1 + i0b*strideCb_M0b;
                        C_out[j0_c+j1*N0][i2*M1*M0 + i1_c*M0 + i0a_c + 2*i0b] = CC[pos1];
                    }
                }
            }
            __syncthreads(); // OK il faut attendre que tout soit chargé.
            // ecriture du Bloc en RAM: suivant M...
            hipint_t i0 = ID_THREAD%(M0*M1*M2); //
            hipint_t j0 = ID_THREAD/(M0*M1*M2); // NB_THREAD = M0*M1*2 / M0*M1*M2
            for (hipint_t j1=0; j1<N0*N1;    j1+=M2>2?1:2/M2) {
                float32_t* __restrict__ C = &c[i4*M0*M1*M2*M3 + i3*M0*M1*M2 + j2*N0*N1*lc + (j1+j0)*lc];
                if (j2*N0*N1+j1+j0<N) {
                    for (hipint_t i1=0; i1<M0*M1*M2; i1+=NB_THREAD) {
                        C[i0+i1] = C_out[j1+j0][i1+i0];
                    }
                }
            }
            __syncthreads(); // OK il faut attendre que tout soit ecrit.
        }
    }

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
    static hip_cache<bfloat16_t> B_cache;
    static hip_cache<float32_t>  C_cache;

    // la fonction hote
    template<std::size_t M1, std::size_t N1, std::size_t M2=1, std::size_t M4=1, std::size_t K1=256>
    void sgemm_wmma(const bfloat16_t* A, const float32_t* B, float32_t* C, std::size_t M, std::size_t N, std::size_t K, std::size_t lb, std::size_t lc) {
        constexpr int M0=16;
        constexpr int N0=16;
        constexpr int K0=16;

        GGML_ASSERT(M%(M0*M1*M2*M4) == 0); // bloc parfait pour M
        GGML_ASSERT(K%(K1) == 0);          // bloc parfait pour K
        GGML_ASSERT(K1%K0 == 0);

        GGML_ASSERT(M*N < 0x80000000);
        GGML_ASSERT(K*N < 0x80000000);
        GGML_ASSERT(M*K < 0x80000000);
        //GGML_ASSERT(M  < 0x8000);
        //GGML_ASSERT(N  < 0x8000);
        //GGML_ASSERT(K  < 0x8000);
        //GGML_ASSERT(lb < 0x8000);
        //GGML_ASSERT(lc < 0x8000);

        // Taille des caches:
        if (B_cache.ensure_size(K1 * block_size<N0*N1>(N) * M4)) {
            IGPU_TRACE("B_cache[" << block_size<M0*M1*M2>(M) << "," << block_size<N1*N0>(N) <<"]");
        }
        if (C_cache.ensure_size(block_size<N1*N0>(N) * block_size<M0*M1*M2>(M))) {
            IGPU_TRACE("C_cache[" << block_size<M0*M1*M2>(M) << "," << block_size<N1*N0>(N) <<"]");
        }

        hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matmul<M2,M1,N1,K1/K0>),
                dim3(M4,nb_block<N0*N1>(N),1), dim3(16, 2, M1),
                0, 0,
                A,B,C, (hipint_t)M,(hipint_t)N,(hipint_t)K,
                B_cache.m_data, (hipint_t)lb,  // lb ~ K
                C_cache.m_data, (hipint_t)lc); // lc ~ M

        auto res = hipGetLastError();
        if (res != hipSuccess) {
            // GGML_LOG_ERROR
            std::cout << " > sgemm_wmma<" <<M<<","<<N<<","<<K<<">"<< std::endl;
            GGML_ABORT("HIP<hipLaunchKernelGGL> failed: %s\n", hipGetErrorString(res));
        }
        HIP_CHECK_ERROR(hipDeviceSynchronize());
    }

}
namespace ggml::backend::igpu::op_mul_mat {

    template<typename TA, typename TB, typename TC> constexpr bool supported_op() { return false; }
    template<> constexpr bool supported_op<bfloat16_t, float32_t, float32_t>() { return true; }

    template<typename TIN, typename TOUT> constexpr bool supported_repack() { return false; }
    template<> constexpr bool supported_repack<bfloat16_t, bfloat16_t>() { return true; }

    // la config :
    // taille de repacking:
    static constexpr std::size_t BLOC_M0 = 16;
    // static constexpr std::size_t BLOC_N0 = 16;
    static constexpr std::size_t BLOC_K0 = 16;
    static constexpr std::size_t BLOC_K1 = 512;

    //std::size_t K_MAX = 0;
    //std::size_t M_MAX = 0;
    //std::size_t N_MAX = 768; // taille optimale pour 768...

	/*
    bool init_caches() {
        // c'est trop tot pour etre pertinant...
        // if (B_cache.ensure_size(K1 * block_size<N0*N1>(N) * M4)) {
        if (B_cache.ensure_size(BLOC_K1 * 1024)) { // @ optimiser la taille de N < 768 * 16
            IGPU_TRACE("B_cache[" << BLOC_K1 << ", " << 1024 <<"]");
        }
        return true;
    }
    */

    inline bool _supported(const ggml_tensor& A, const ggml_tensor& B, const ggml_tensor& C) {
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

    // pack des poids (A)
    inline bool _repack(const bfloat16_t* ref, std::size_t la, bfloat16_t* bloc, std::size_t M, std::size_t K) {

        // TODO: @ optimiser...
        /*
//#           pragma omp parallel for num_threads(4)
#           pragma omp parallel for
            for (std::size_t i=0; i<M; i++) {
                for (std::size_t k=0; k<K; k++) {
                    bloc[posBloc2D<BLOC_K1,BLOC_M0,1,TYPE_BLOC::PERFECT>(K, M, k, i)] = ref[pos2D(la, M, k, i)];
                }
            }
         */
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
                            bloc[posBloc2D<BLOC_K1,BLOC_M0,1,TYPE_BLOC::PERFECT>(K, M, k2+k1+k0, i1+i0)] = tmp[i0][k0];
                        }
                    }
                    // bloc[posBloc2D<BLOC_K1,BLOC_M0,1,TYPE_BLOC::PERFECT>(K, M, k, i)] = ref[pos2D(la, M, k, i)];
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
        // if (N==1) {} else
        if (N <= 0) {
            return true;
        } else
            if (N<=16) {
                if (M%(4*2*16*16)==0) { // M=2048
                    sgemm_wmma<4,1,2,16,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*8)==0) { // M=1024
                    sgemm_wmma<4,1,2,8,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*4)==0) { // M=512
                    sgemm_wmma<4,1,2,4,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*2)==0) { // M=256
                    sgemm_wmma<4,1,2,2,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*1)==0) { // M=128
                    sgemm_wmma<4,1,2,1,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else { // est-ce que l'on fait les cas 3,5,6,7,... ?
                    // on va s'arreter la pour l'instant:
                    return false;
                }
            } else if (N<=32) {
                if (M%(4*2*16*16)==0) { // M=2048
                    sgemm_wmma<4,2,2,16,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*8)==0) { // M=1024
                    sgemm_wmma<4,2,2,8,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*4)==0) { // M=512
                    sgemm_wmma<4,2,2,4,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*2)==0) { // M=256
                    sgemm_wmma<4,2,2,2,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*1)==0) { // M=128
                    sgemm_wmma<4,2,2,1,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else { // est-ce que l'on fait les cas 3,5,6,7,... ?
                    // on va s'arreter la pour l'instant:
                    return false;
                }
            } else if (N<=48) { // 3 blocs pour N => 4CU / M
                if (M%(4*2*16*8)==0) { // M=1024
                    sgemm_wmma<4,1,2,8,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*4)==0) { // M=512
                    sgemm_wmma<4,1,2,4,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*2)==0) { // M=256
                    sgemm_wmma<4,1,2,2,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*1)==0) { // M=128
                    sgemm_wmma<4,1,2,1,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else { // est-ce que l'on fait les cas 3,5,6,7,... ?
                    // on va s'arreter la pour l'instant:
                    return false;
                }
            } else if (N<=64) { // N1=2 => 2CU/N => 6 restant
                if (M%(4*2*16*8)==0) { // M=1024
                    sgemm_wmma<4,2,2,8,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*4)==0) { // M=512
                    sgemm_wmma<4,2,2,4,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*2)==0) { // M=256
                    sgemm_wmma<4,2,2,2,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*1)==0) { // M=128
                    sgemm_wmma<4,2,2,1,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else { // est-ce que l'on fait les cas 3,5,6,7,... ?
                    // on va s'arreter la pour l'instant:
                    return false;
                }
            } else if (N<=192) {
                if (M%(4*2*16*4)==0) { // M=512
                    sgemm_wmma<4,2,2,4,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*2)==0) { // M=256
                    sgemm_wmma<4,2,2,2,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*1)==0) { // M=128
                    sgemm_wmma<4,2,2,1,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else { // est-ce que l'on fait les cas 3,5,6,7,... ?
                    // on va s'arreter la pour l'instant:
                    return false;
                }
            } else if (N<=384) {
                if (M%(4*2*16*2)==0) { // M=256
                    sgemm_wmma<4,2,2,2,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else if (M%(4*2*16*1)==0) { // M=128
                    sgemm_wmma<4,2,2,1,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else { // est-ce que l'on fait les cas 3,5,6,7,... ?
                    // on va s'arreter la pour l'instant:
                    return false;
                }
            } else {
                if (M%(4*2*16*1)==0) { // M=128
                    sgemm_wmma<4,2,2,1,BLOC_K1>(A, B, C, M,N,K, lb,lc);
                } else { // est-ce que l'on fait les cas 3,5,6,7,... ?
                    // on va s'arreter la pour l'instant:
                    return false;
                }
            }
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
