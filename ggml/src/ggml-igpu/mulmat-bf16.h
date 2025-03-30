// @ builder avec HIP...

#include "ggml-hip.h"
#include <cstddef>

//#if 1
#if 0
#define __host__
#define __device__
#define __global__
#define __shared__
#define __launch_bounds__(...)
#endif

namespace ggml::backend::igpu {

    // les types...
    using bfloat16_t = __bf16;
    using float32_t  = float;

    using bfloat16x16_t = __bf16 __attribute__((ext_vector_type(16)));
    using float32x8_t   =  float __attribute__((ext_vector_type(8)));

    template<typename T> __device__ bfloat16_t conv2bf16(T val);
    template<> __device__ inline bfloat16_t conv2bf16(float32_t val) {
        // ca semble suffisant => peut-etre "generique"?
        return val;
    }
    template<> __device__ inline bfloat16_t conv2bf16(bfloat16_t val) { return val; }

    // les kernel coté hote.
    // - load A/B matrice from RAM
    template <std::size_t NB, std::size_t SIZE, std::size_t K0, typename T>
    __device__ inline void load(const T* X, bfloat16_t X_frag[SIZE][K0+2],
            std::size_t I0, std::size_t i0, std::size_t k0, std::size_t k, std::size_t lx, std::size_t N)
    {
        // on copie: X[i0:i0+NB, k0:k0+K0] => X_Flag[I0:I0+NB, 0:K0];
        for (int i=0; i<NB; ++i) {
            if ((i0+i)<N) {
                X_frag[I0+i][k] = conv2bf16(X[lx*(i0+i) + k0+k]);
            } else {
                X_frag[I0+i][k] = 0;
            }
        }
    }
    template <std::size_t NB, std::size_t SIZE, std::size_t K0, typename T>
    __device__ inline void load(const T* X, bfloat16_t X_frag[SIZE][K0+2],
            std::size_t I0, std::size_t i0, std::size_t k0, std::size_t k, std::size_t lx)
    {
        for (int i=0; i<NB; ++i) {
            X_frag[I0+i][k] = conv2bf16(X[lx*(i0+i) + k0+k]);
        }
    }
    // - produit matriciel:
    // > cas ou ( M % (M2*M1*M0) == 0 )
    template<std::size_t M2, std::size_t N2, std::size_t M1, std::size_t N1,
    std::size_t M0=16, std::size_t N0=16, std::size_t K0=16,
    typename TA, typename TB>
    __global__ void __launch_bounds__(16*2*N2*M2) wmma_matmul(
            const TA* __restrict__ a, const TB* __restrict__ b, float32_t* __restrict__ c,
            std::size_t M, std::size_t N, std::size_t K,
            std::size_t la, std::size_t lb, std::size_t lc
            )
    {
        // only possible values!
        static_assert(M0==16);
        static_assert(N0==16);
        static_assert(K0==16);

        //  block:  C[i0:i0+M*16,j0:j0+N*16];
        const int I0 = blockIdx.x*M0*M1*M2;
        const int J0 = blockIdx.y*N0*N1*N2;
        const int lane = threadIdx.x;    // == M0/N0/K0  [0..16]
        const int wave = threadIdx.y;    // 0 ou 1 suivant si on trait les paires ou les impaires.
        // threadIdx.z  € [0..M2*N2]
        const int i2 = (threadIdx.z%M2) * M0*M1;  // [0..M1]*16*M1
        const int j2 = (threadIdx.z/M2) * N0*N1;  // [0..N1]*16*N1

        // strategie: lane<=>k ; les autre sont a repartir sur [0..M1*N1]
        constexpr int M_size    = M0*M1*M2;
        constexpr int N_size    = N0*N1*N2;

        constexpr int NB_BLOC   = 2*M2*N2; // == nb_tread/nb_lane(16)/2 wave...
        constexpr int BLOC_A    = M_size/NB_BLOC;  static_assert(M_size%NB_BLOC == 0);  // la taille du bloc (/lane)
        constexpr int BLOC_B    = N_size/NB_BLOC;  static_assert(N_size%NB_BLOC == 0);
        const int IA = (2*threadIdx.z+wave)*BLOC_A;
        const int JB = (2*threadIdx.z+wave)*BLOC_B;

        __shared__ bfloat16_t A_frag[2][M_size][K0+2];  // [flip/flop][I][K]
        __shared__ bfloat16_t B_frag[2][N_size][K0+2];  // [flip/flop][J][K]

        // initialize c fragment to 0
        float32x8_t   c_frag[M1][N1] = {};  // [i:i+16]
        bfloat16x16_t a_frag[M1]; // [k..k+K0]
        bfloat16x16_t b_frag[N1]; // [k..k+K0]

        // chargement des elements en "local"
        //load<BLOC_A,M_size,K0>(a, A_frag[0], IA, I0+IA, 0, lane, la, M);
        load<BLOC_A,M_size,K0>(a, A_frag[0], IA, I0+IA, 0, lane, la);
        load<BLOC_B,N_size,K0>(b, B_frag[0], JB, J0+JB, 0, lane, lb, N);
        __syncthreads();

        int l1=K0;
        int flip=0;
        int flop=1;

        for (; l1<K; l1+=K0) {

            for (std::size_t _i1=0; _i1<M1; ++_i1) {
                #pragma unrool K0
                for (std::size_t _k=0; _k<K0; _k++) a_frag[_i1][_k]=A_frag[flip][i2+_i1*M0+lane][_k];
            }
            for (std::size_t _j1=0; _j1<N1; ++_j1) {
                const std::size_t J = (J0+j2+_j1*N0);
                #pragma unrool K0
                for (std::size_t _k=0; _k<K0; _k++) b_frag[_j1][_k]=B_frag[flip][j2+_j1*N0+lane][_k];
            }
            // les 2 wave remplisent 1/2 de c!
            for (std::size_t _i1=0; _i1<M1; ++_i1) {
                #pragma unrool N1
                for (std::size_t _j1=0; _j1<N1; ++_j1) {
                    c_frag[_i1][_j1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag[_i1], b_frag[_j1], c_frag[_i1][_j1]);
                }
            }

            flop = (flip+1)%2;
            //load<BLOC_A,M_size,K0>(a, A_frag[flop], IA, I0+wave*BLOC_A, l1, la, M);
            load<BLOC_A,M_size,K0>(a, A_frag[flop], IA, I0+IA, l1, lane, la);
            load<BLOC_B,N_size,K0>(b, B_frag[flop], JB, J0+JB, l1, lane, lb, N);
            flip = flop;
            __syncthreads();
        }
        for (std::size_t _i1=0; _i1<M1; ++_i1) {
            #pragma unrool K0
            for (int _k=0; _k<K0; _k++) a_frag[_i1][_k]=A_frag[flip][i2+_i1*M0+lane][_k];
        }
        for (std::size_t _j1=0; _j1<N1; ++_j1) {
            #pragma unrool K0
            for (int _k=0; _k<K0; _k++) b_frag[_j1][_k]=B_frag[flip][j2+_j1*N0+lane][_k];
        }

        for (std::size_t _i1=0; _i1<M1; ++_i1) {
            for (std::size_t _j1=0; _j1<N1; ++_j1) {
                c_frag[_i1][_j1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag[_i1], b_frag[_j1], c_frag[_i1][_j1]);
            }
        }

        // ecriture de C
        for (std::size_t _i1=0; _i1<M1; ++_i1) {
            // #pragma unrool
            for (std::size_t _j1=0; _j1<N1; ++_j1) {
                const std::size_t J = (J0+j2+_j1*N0+lane);
                const std::size_t I = I0+i2+_i1*M0;
                if (J<N) {
                    const std::size_t pos = lc*J + I;
                    #pragma unrool
                    for (std::size_t ele = 0; ele < M0/2; ++ele) { // == i
                        if ((I + ele*2 + wave)<M) {
                            c[ pos + ele*2 + wave] = c_frag[_i1][_j1][ele];
                        }
                    }
                }
            }
        }
    }

    // les kernel coté hote.
    template<int M2, int N2, int M1, int N1, typename TA, typename TB>
    __host__ inline void sgemm_wmma(TA* a, TB* b, float32_t* c,
            std::size_t  M, std::size_t  N, std::size_t  K,
            std::size_t la, std::size_t lb, std::size_t lc
            )
    {
        constexpr int M0=16;
        constexpr int N0=16;
        constexpr int K0=16;

        std::size_t nbBlockM = (M-1)/(M0*M1*M2)+1;
        std::size_t nbBlockN = (N-1)/(N0*N1*N2)+1;

        hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matmul<M2,N2,M1,N1>),
                           dim3(nbBlockM,nbBlockN,1), dim3(16, 2, M2*N2),
                           0, 0,
                           a,b,c, M,N,K, la,lb,lc);

        auto res = hipGetLastError();
        if (res != hipSuccess) {
            GGML_ABORT("HIP<hipLaunchKernelGGL> failed: %s\n", hipGetErrorString(res));
        }
        HIP_CHECK_ERROR(hipDeviceSynchronize());
    }

}
