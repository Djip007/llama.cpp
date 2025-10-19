#include "ops.h"

#include "hip-tools.h"
#include "hip-type.h"

namespace ggml::backend::igpu::op {
namespace binary_imp {

//======================
// les helper:
template<typename TIN0, typename TIN1, typename TOUT, ggml_op OP>
struct op_compute {
    static constexpr bool implemented = false;
};

enum class type_imp {
   NOT_IMPLEMENTED,
   NATIVE,
   SIMPLE,
   SIMPLE_CONVERT
   //... etc.
};
template<ggml_type IN0, ggml_type IN1, ggml_type OUT, ggml_op OP> struct op_config {
    static constexpr type_imp implementation = type_imp::NOT_IMPLEMENTED;
};

//======================
// implementations:
//  - les OP
template<> struct op_compute<float32_t, float32_t, float32_t, GGML_OP_ADD> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x, float32_t y) { return x+y; }
};
template<> struct op_compute<float32_t, float32_t, float32_t, GGML_OP_MUL> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x, float32_t y) { return x*y; }
};
template<> struct op_compute<float32_t, float32_t, float32_t, GGML_OP_SUB> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x, float32_t y) { return x-y; }
};
template<> struct op_compute<float32_t, float32_t, float32_t, GGML_OP_DIV> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x, float32_t y) { return x/y; }
};

//======================
//  - les config
// > cas natif float32
template<ggml_op OP> struct op_config<GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, OP> {
    using op = op_compute<float32_t, float32_t, float32_t, OP>;
    static constexpr type_imp implementation = op::implemented ? type_imp::NATIVE : type_imp::NOT_IMPLEMENTED;
};
// - cas simple avec conversion en fp32:
template<ggml_op OP> struct op_config<GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F16, OP> {
    using op = op_compute<float32_t, float32_t, float32_t, OP>;
    static constexpr type_imp implementation = op::implemented ? type_imp::SIMPLE : type_imp::NOT_IMPLEMENTED;
    using compute_type_in0 = float32_t;
    using compute_type_in1 = float32_t;
    using compute_type_out = float32_t;
};
template<ggml_op OP> struct op_config<GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_BF16, OP> {
    using op = op_compute<float32_t, float32_t, float32_t, OP>;
    static constexpr type_imp implementation = op::implemented ? type_imp::SIMPLE : type_imp::NOT_IMPLEMENTED;
    using compute_type_in0 = float32_t;
    using compute_type_in1 = float32_t;
    using compute_type_out = float32_t;
};

//======================
// - les kernel HIP et leurs usages generique
template<ggml_type IN0, ggml_type IN1, ggml_type OUT, ggml_op OP>
__global__ void kernel(const typename to_type<IN0>::type * in0, const typename to_type<IN0>::type * in1, typename to_type<OUT>::type * out,
                       const int N0, const int N1, const int N2, const int N3, // les dim de in0/out
                       const int M0, const int M1, const int M2, const int M3  // les dim de in1
) {
    // les indices sur in0 / out
    const auto i0 = blockDim.x*blockIdx.x + threadIdx.x;
    const auto i1 = blockDim.y*blockIdx.y + threadIdx.y;
    const auto  z = blockDim.z*blockIdx.z + threadIdx.z;
    const auto i2 = z % N2;
    const auto i3 = z / N2;
    if (i0 >= N0) { return; }
    if (i1 >= N1) { return; }
    if (i2 >= N2) { return; }
    if (i3 >= N3) { return; }
    // les indices sur in1:
    const auto j0 = i0 % M0;
    const auto j1 = i1 % M1;
    const auto j2 = i2 % M2;
    const auto j3 = i3 % M3;
    // les indices dans les matrices:
    const auto i = i0 + i1*N0 + i2*N1*N0 + i3*N2*N1*N0;
    const auto j = j0 + j1*M0 + j2*M1*M0 + j3*M2*M1*M0;

    using config = op_config<IN0, IN1, OUT, OP>;
    // cas types natifs
    if constexpr(config::implementation == type_imp::NATIVE) {
        using in0_t = to_type<IN0>::type;
        using in1_t = to_type<IN1>::type;
        using out_t = to_type<OUT>::type;
        op_compute<in0_t, in1_t, out_t, OP> op;
        out[i] = op.compute(in0[i], in1[j]);
    }
    // cas simple avec conversion...
    if constexpr(config::implementation == type_imp::SIMPLE) {
        using in0_t = to_type<IN0>::type;
        using in1_t = to_type<IN1>::type;
        using out_t = to_type<OUT>::type;
        using cin0_t = config::compute_type_in0;
        using cin1_t = config::compute_type_in1;
        using cout_t = config::compute_type_out;
        op_compute<cin0_t, cin1_t, cout_t, OP> op;
        type::to<in0_t,cin0_t> conv_in0;
        type::to<in1_t,cin1_t> conv_in1;
        type::to<cout_t,out_t> conv_out;
        out[i] = conv_out(op.compute(conv_in0(in0[i]),conv_in1(in1[j])));
    }
}

template<ggml_type IN0, ggml_type IN1, ggml_type OUT, ggml_op OP>
struct op_imp {
    using config = op_config<IN0, IN1, OUT, OP>;
    bool supported(const ggml_tensor & op, const ggml_tensor & in0, const ggml_tensor & in1, const ggml_tensor & out) {
        if constexpr(config::implementation != type_imp::NOT_IMPLEMENTED) {
            bool allowed = true;
            // qq hypotheses pour commancer
            // - contigus
            allowed &= ggml_is_contiguous(&in0);
            allowed &= ggml_is_contiguous(&in1);
            allowed &= ggml_is_contiguous(&out);
            // - broadcast simple:
            for (int i=0; i<GGML_MAX_DIMS; ++i) {
                //allowed &= in0.nb[i] == in1.nb[i] || in1.nb[i] == 1;
                allowed &= (in0.nb[i] % in1.nb[i]) == 0;
            }
            // - meme taille entre in0 et out...
            allowed &= ggml_are_same_shape(&in0, &out);
            return allowed;
        }
        return false;
    }
    bool compute(const ggml_tensor & op, const ggml_tensor & in0, const ggml_tensor & in1, ggml_tensor & out) {
        if constexpr(config::implementation != type_imp::NOT_IMPLEMENTED) {
            // il faut dispacher sur x,y,z  => quel taille de bloc?
            using in0_t = to_type<IN0>::type;
            using in1_t = to_type<IN1>::type;
            using out_t = to_type<OUT>::type;
            const auto N0 = in0.ne[0];
            const auto N1 = in0.ne[1];
            const auto N2 = in0.ne[2];
            const auto N3 = in0.ne[3];
            const auto M0 = in1.ne[0];
            const auto M1 = in1.ne[1];
            const auto M2 = in1.ne[2];
            const auto M3 = in1.ne[3];
            constexpr int BLOCK_SIZE = 256;
            const int num_blocks_x = (N0 - 1) / BLOCK_SIZE + 1;
            const int num_blocks_y = N1;
            const int num_blocks_z = N2*N3;
            kernel<IN0, IN1, OUT, OP><<<dim3(num_blocks_x,num_blocks_y,num_blocks_z), dim3(BLOCK_SIZE,1,1), 0>>>(
                (const in0_t*)in0.data, (const in1_t*)in1.data, (out_t*)out.data, N0,N1,N2,N3, M0,M1,M2,M3);
            return true;
        }
        return false;
    }
};

//======================
//  - les mapage
enum fct_t {
    SUPPORT,
    COMPUTE
};

template<fct_t FCT, ggml_op OP, ggml_type IN0, ggml_type IN1, ggml_type OUT,
         typename T1, typename T2, typename T3, typename T4>
bool fct_op_in0_in1_out(T1 & op, T2 & in0, T3 & in1, T4 & out) {
    op_imp<IN0,IN1,OUT,OP> imp;
    if constexpr(FCT == SUPPORT) {
        return imp.supported(op, in0, in1, out);
    }
    if constexpr(FCT == COMPUTE) {
        return imp.compute(op, in0, in1, out);
    }
    return false;
}

template<fct_t FCT, ggml_op OP, ggml_type IN0, ggml_type IN1,
         typename T1, typename T2, typename T3, typename T4>
bool fct_op_in0_in1(T1 & op, T2 & in0, T3 & in1, T4 & out) {
    switch(in1.type) {
        case GGML_TYPE_F32  : return fct_op_in0_in1_out<FCT, OP, IN0, IN1, GGML_TYPE_F32 >(op, in0, in1, out);
        case GGML_TYPE_F16  : return fct_op_in0_in1_out<FCT, OP, IN0, IN1, GGML_TYPE_F16 >(op, in0, in1, out);
        case GGML_TYPE_BF16 : return fct_op_in0_in1_out<FCT, OP, IN0, IN1, GGML_TYPE_BF16>(op, in0, in1, out);
        default: return false;
    }
    return false;
}

template<fct_t FCT, ggml_op OP, ggml_type IN0,
         typename T1, typename T2, typename T3, typename T4>
bool fct_op_in0(T1 & op, T2 & in0, T3 & in1, T4 & out) {
    switch(in1.type) {
        case GGML_TYPE_F32  : return fct_op_in0_in1<FCT, OP, IN0, GGML_TYPE_F32 >(op, in0, in1, out);
        case GGML_TYPE_F16  : return fct_op_in0_in1<FCT, OP, IN0, GGML_TYPE_F16 >(op, in0, in1, out);
        case GGML_TYPE_BF16 : return fct_op_in0_in1<FCT, OP, IN0, GGML_TYPE_BF16>(op, in0, in1, out);
        default: return false;
    }
    return false;
}

template<fct_t FCT, ggml_op OP,
         typename T1, typename T2, typename T3, typename T4>
bool fct_op(T1 & op, T2 & in0, T3 & in1, T4 & out) {
    switch(in0.type) {
        case GGML_TYPE_F32  : return fct_op_in0<FCT, OP, GGML_TYPE_F32 >(op, in0, in1, out);
        case GGML_TYPE_F16  : return fct_op_in0<FCT, OP, GGML_TYPE_F16 >(op, in0, in1, out);
        case GGML_TYPE_BF16 : return fct_op_in0<FCT, OP, GGML_TYPE_BF16>(op, in0, in1, out);
        default: return false;
    }
    return false;
}

template<fct_t FCT,
         typename T1, typename T2, typename T3, typename T4>
bool fct(T1 & op, T2 & in0, T3 & in1, T4 & out) {
    switch (op.op) {
        case GGML_OP_ADD: return fct_op<FCT, GGML_OP_ADD>(op, in0, in1, out);
        case GGML_OP_MUL: return fct_op<FCT, GGML_OP_MUL>(op, in0, in1, out);
        case GGML_OP_SUB: return fct_op<FCT, GGML_OP_SUB>(op, in0, in1, out);
        case GGML_OP_DIV: return fct_op<FCT, GGML_OP_DIV>(op, in0, in1, out);
        default: return false;
    }
}
}

bool binary::supports(const ggml_tensor & op, const ggml_tensor & in0, const ggml_tensor & in1, const ggml_tensor & out) {
    return binary_imp::fct<binary_imp::SUPPORT>(op, in0, in1, out);
}
void binary::compute(const ggml_tensor & op, const ggml_tensor & in0, const ggml_tensor & in1, ggml_tensor & out) {
    binary_imp::fct<binary_imp::COMPUTE>(op, in0, in1, out);
}
}
