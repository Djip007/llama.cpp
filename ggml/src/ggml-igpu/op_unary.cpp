#include "ops.h"

#include "hip-tools.h"
#include "hip-type.h"

static constexpr bool NOT_IMPLEMENTED = true;
namespace ggml::backend::igpu::op {
namespace unary_imp {

//======================
// les helper:
template<typename TIN, typename TOUT, ggml_op OP, ggml_unary_op UOP>
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
template<ggml_type IN, ggml_type OUT, ggml_op OP, ggml_unary_op UOP> struct op_config {
    static constexpr type_imp implementation = type_imp::NOT_IMPLEMENTED;
};

//======================
// implementations:
//  - les OP
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_ABS> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return fabsf(x); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_SGN> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f))); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_NEG> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return -x; }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_STEP> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return (x > 0.f) ? 1.f : 0.f; }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_TANH> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return tanhf(x); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_ELU> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return (x > 0.f) ? x : expm1f(x); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_RELU> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return fmaxf(x, 0); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_SIGMOID> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return 1.0f / (1.0f + expf(-x)); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_GELU> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) {
        constexpr float GELU_COEF_A    = 0.044715f;
        constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
    }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_GELU_QUICK> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) {
        constexpr float GELU_QUICK_COEF = -1.702f;
        return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
    }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_SILU> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return x / (1.0f + expf(-x)); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_HARDSWISH> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f)); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_HARDSIGMOID> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f)); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_EXP> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return expf(x); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_UNARY, GGML_UNARY_OP_GELU_ERF> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) {
        constexpr float SQRT_2_INV = 0.70710678118654752440084436210484f;
        return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
    }
};
// TODO: GGML_OP_UNARY, GGML_UNARY_OP_XIELU      @ voir pour ce dernier

//   + celle qui se comportent pareil (GGML_UNARY_OP_COUNT: fictif)
template<> struct op_compute<float32_t, float32_t, GGML_OP_SQR, GGML_UNARY_OP_COUNT> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return x*x; }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_SQRT, GGML_UNARY_OP_COUNT> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return sqrtf(x); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_LOG, GGML_UNARY_OP_COUNT> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return logf(x); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_SIN, GGML_UNARY_OP_COUNT> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return sinf(x); }
};
template<> struct op_compute<float32_t, float32_t, GGML_OP_COS, GGML_UNARY_OP_COUNT> {
    static constexpr bool implemented = true;
    __device__ __forceinline__ float32_t compute(float32_t x) { return cosf(x); }
};

//======================
//  - les config
// > cas natif float32
template<ggml_op OP, ggml_unary_op UOP> struct op_config<GGML_TYPE_F32, GGML_TYPE_F32, OP, UOP> {
    using op = op_compute<float32_t, float32_t, OP, UOP>;
    static constexpr type_imp implementation = op::implemented ? type_imp::NATIVE : type_imp::NOT_IMPLEMENTED;
};
// - cas simple avec conversion en fp32:
template<ggml_op OP, ggml_unary_op UOP> struct op_config<GGML_TYPE_F16, GGML_TYPE_F16, OP, UOP> {
    using op = op_compute<float32_t, float32_t, OP, UOP>;
    static constexpr type_imp implementation = op::implemented ? type_imp::SIMPLE : type_imp::NOT_IMPLEMENTED;
    using compute_type_in  = float32_t;
    using compute_type_out = float32_t;
};
template<ggml_op OP, ggml_unary_op UOP> struct op_config<GGML_TYPE_BF16, GGML_TYPE_BF16, OP, UOP> {
    using op = op_compute<float32_t, float32_t, OP, UOP>;
    static constexpr type_imp implementation = op::implemented ? type_imp::SIMPLE : type_imp::NOT_IMPLEMENTED;
    using compute_type_in  = float32_t;
    using compute_type_out = float32_t;
};
// - cas simple avec type de sortie diferrent pour fusions d'OP:
template<ggml_op OP, ggml_unary_op UOP> struct op_config<GGML_TYPE_F32, GGML_TYPE_F16, OP, UOP> {
    using op = op_compute<float32_t, float32_t, OP, UOP>;
    static constexpr type_imp implementation = op::implemented ? type_imp::SIMPLE : type_imp::NOT_IMPLEMENTED;
    using compute_type_in  = float32_t;
    using compute_type_out = float32_t;
};
template<ggml_op OP, ggml_unary_op UOP> struct op_config<GGML_TYPE_F32, GGML_TYPE_BF16, OP, UOP> {
    using op = op_compute<float32_t, float32_t, OP, UOP>;
    static constexpr type_imp implementation = op::implemented ? type_imp::SIMPLE : type_imp::NOT_IMPLEMENTED;
    using compute_type_in  = float32_t;
    using compute_type_out = float32_t;
};

// si besoin il est tjs possible de specialiser plus finement.
// [...]

//======================
// - les kernel HIP et leurs usages generique
template<ggml_type IN, ggml_type OUT, ggml_op OP, ggml_unary_op UOP>
__global__ void kernel(const typename to_type<IN>::type * in, typename to_type<OUT>::type * out, const int k) {
    const auto i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) { return; }

    using config = op_config<IN,OUT,OP,UOP>;
    // cas simple avec conversion...
    if constexpr(config::implementation == type_imp::SIMPLE) {
        using in_t  = to_type<IN>::type;
        using out_t = to_type<OUT>::type;
        using cin_t  = config::compute_type_in;
        using cout_t = config::compute_type_out;
        op_compute<cin_t, cout_t, OP, UOP> op;
        type::to<in_t,cin_t> conv_in;
        type::to<cout_t,out_t> conv_out;
        out[i] = conv_out(op.compute(conv_in(in[i])));
    }
    // cas types natifs
    if constexpr(config::implementation == type_imp::NATIVE) {
        using in_t  = to_type<IN>::type;
        using out_t = to_type<OUT>::type;
        op_compute<in_t, out_t, OP, UOP> op;
        out[i] = op.compute(in[i]);
    }
    // autres..... bloc/simple...
}

template<ggml_type IN, ggml_type OUT, ggml_op OP, ggml_unary_op UOP>
struct op_imp {
    using config = op_config<IN, OUT, OP, UOP>;

    bool supported(const ggml_tensor & op, const ggml_tensor & in, const ggml_tensor & out) {
        if constexpr(config::implementation != type_imp::NOT_IMPLEMENTED) {
            return ggml_is_contiguous(&in) && ggml_is_contiguous(&out);
            // TODO voir a traiter les autres cas...
        }
        return false;
    }

    bool compute(const ggml_tensor & op, const ggml_tensor & in, ggml_tensor & out) {
        // @ voir plusieurs strategie suivant les types QK / FP / ...
        if constexpr(config::implementation != type_imp::NOT_IMPLEMENTED) {
            using in_t  = to_type<IN>::type;
            using out_t = to_type<OUT>::type;
            auto k = ggml_nelements(&in); // 1/nb_element_par_bloc pour les quantisés
            if (k>0) {
                constexpr int BLOCK_SIZE = 256;
                const int num_blocks = (k - 1) / BLOCK_SIZE + 1;
                kernel<IN, OUT, OP, UOP><<<num_blocks, BLOCK_SIZE, 0>>>((const in_t*)in.data, (out_t*)out.data, k);
            }
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

template<fct_t FCT, ggml_op OP, ggml_unary_op UOP, ggml_type IN, ggml_type OUT, typename T1, typename T2, typename T3>
bool fct_op_in_out(T1 & op, T2 & in, T3 & out) {
    op_imp<IN,OUT,OP,UOP> imp;
    if constexpr(FCT == SUPPORT) {
        return imp.supported(op, in, out);
    }
    if constexpr(FCT == COMPUTE) {
        return imp.compute(op, in, out);
    }
    return false;
}

template<fct_t FCT, ggml_op OP, ggml_unary_op UOP, ggml_type IN, typename T1, typename T2, typename T3>
bool fct_op_in(T1 & op, T2 & in, T3 & out) {
    switch(out.type) {
        case GGML_TYPE_F32  : return fct_op_in_out<FCT, OP, UOP, IN, GGML_TYPE_F32 >(op, in, out);
        case GGML_TYPE_F16  : return fct_op_in_out<FCT, OP, UOP, IN, GGML_TYPE_F16 >(op, in, out);
        case GGML_TYPE_BF16 : return fct_op_in_out<FCT, OP, UOP, IN, GGML_TYPE_BF16>(op, in, out);
        default: return false;
    }
    return false;
}

template<fct_t FCT, ggml_op OP, ggml_unary_op UOP, typename T1, typename T2, typename T3>
bool fct_op(T1 & op, T2 & in, T3 & out) {
    switch(in.type) {
        case GGML_TYPE_F32  : return fct_op_in<FCT, OP, UOP, GGML_TYPE_F32 >(op, in, out);
        case GGML_TYPE_F16  : return fct_op_in<FCT, OP, UOP, GGML_TYPE_F16 >(op, in, out);
        case GGML_TYPE_BF16 : return fct_op_in<FCT, OP, UOP, GGML_TYPE_BF16>(op, in, out);
        default: return false;
    }
    return false;
}

template<fct_t FCT, typename T1, typename T2, typename T3>
bool fct(T1 & op, T2 & in, T3 & out) {
    switch (op.op) {
        case GGML_OP_UNARY:
		    switch (ggml_get_unary_op(&op)) {
		        case GGML_UNARY_OP_ABS:         return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_ABS        >(op, in, out);
		        case GGML_UNARY_OP_SGN:         return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_SGN        >(op, in, out);
		        case GGML_UNARY_OP_NEG:         return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_NEG        >(op, in, out);
		        case GGML_UNARY_OP_STEP:        return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_STEP       >(op, in, out);
		        case GGML_UNARY_OP_TANH:        return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_TANH       >(op, in, out);
		        case GGML_UNARY_OP_ELU:         return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_ELU        >(op, in, out);
		        case GGML_UNARY_OP_RELU:        return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_RELU       >(op, in, out);
		        case GGML_UNARY_OP_SIGMOID:     return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_SIGMOID    >(op, in, out);
		        case GGML_UNARY_OP_GELU:        return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_GELU       >(op, in, out);
		        case GGML_UNARY_OP_GELU_QUICK:  return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_GELU_QUICK >(op, in, out);
		        case GGML_UNARY_OP_SILU:        return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_SILU       >(op, in, out);
		        case GGML_UNARY_OP_HARDSWISH:   return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_HARDSWISH  >(op, in, out);
		        case GGML_UNARY_OP_HARDSIGMOID: return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_HARDSIGMOID>(op, in, out);
		        case GGML_UNARY_OP_EXP:         return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_EXP        >(op, in, out);
		        case GGML_UNARY_OP_GELU_ERF:    return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_GELU_ERF   >(op, in, out);
		        case GGML_UNARY_OP_XIELU:       return fct_op<FCT, GGML_OP_UNARY, GGML_UNARY_OP_XIELU      >(op, in, out);
		        // pour ne pas avoir de warning, ce n'est pas une OP
		        case GGML_UNARY_OP_COUNT:       return false;
		    }
            break;
        case GGML_OP_SQR:  return fct_op<FCT, GGML_OP_SQR , GGML_UNARY_OP_COUNT>(op, in, out);
        case GGML_OP_SQRT: return fct_op<FCT, GGML_OP_SQRT, GGML_UNARY_OP_COUNT>(op, in, out);
        case GGML_OP_LOG:  return fct_op<FCT, GGML_OP_LOG , GGML_UNARY_OP_COUNT>(op, in, out);
        case GGML_OP_SIN:  return fct_op<FCT, GGML_OP_SIN , GGML_UNARY_OP_COUNT>(op, in, out);
        case GGML_OP_COS:  return fct_op<FCT, GGML_OP_COS , GGML_UNARY_OP_COUNT>(op, in, out);
        default: return false;
    }
    return false;
}

} // namespace unary_imp

// - les methodes publiques:
bool unary::supports(const ggml_tensor & op, const ggml_tensor & in, const ggml_tensor & out) {
    return unary_imp::fct<unary_imp::SUPPORT>(op, in, out);
}
void unary::compute(const ggml_tensor & op, const ggml_tensor & in, ggml_tensor & out) {
    unary_imp::fct<unary_imp::COMPUTE>(op, in, out);
}
}