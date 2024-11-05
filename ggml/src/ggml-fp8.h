// @ revoir  plutot un FQ8_N  FQ8_E3M4 / FQ8_E4M3 ... sans denorme ...
#include "ggml.h"
#include "ggml-common.h"

// this is more a .inc.
#ifdef  __cplusplus

#include <cstdint>

template<int N>
constexpr int exp_i2() {
    return 1 << N;
}

template<int N>
constexpr float exp_f2() {
    if constexpr (N>0)  {return exp_f2<N-1>()*2;}
    if constexpr (N<0)  {return exp_f2<N+1>()/2;}
    if constexpr (N==0) {return 1.;}
}

template<int _E> //, int M=7-E>  1.7 bits!
struct FP8 {
    std::uint8_t bits;
    using type = FP8<_E>;
    static constexpr int E      = _E;
    static constexpr int M      = (7-_E);
    static constexpr int E_BIAS = exp_i2<E-1>()-1;
    static constexpr float MAX  = (2-exp_f2<-M+1>())*exp_f2<exp_i2<E-1>()>();
    static constexpr float MIN  = exp_f2<-M>()*exp_f2<2-exp_i2<E-1>()>();
};

// + FQ8<E>
template<int _E> //, int M=7-E>  1.7 bits!
struct FQ8 {
    std::uint8_t bits;
    using type = FQ8<_E>;
    static constexpr int E      = _E;
    static constexpr int M      = (7-_E);
    // static constexpr int E_BIAS = 0; // on fait un scale donc ca ne change rien... @ supprimer? nombre de [1 ... 2^2^E)
    static constexpr float MAX  = (2-exp_f2<-M>()) * exp_f2<exp_i2<E>()-1>();  // mantice * exposant
    static constexpr float MIN  =  1+exp_f2<-M>(); // [1.0...01]
};

extern "C" {
#endif

    // Note: types are define in ggml-common.h
    GGML_API void dequantize_row_e4m3_q  (const block_e4m3_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void quantize_row_e4m3_q_ref(const float * GGML_RESTRICT x, block_e4m3_q * GGML_RESTRICT y, int64_t k);
    GGML_API void dequantize_row_e3m4_q  (const block_e3m4_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void quantize_row_e3m4_q_ref(const float * GGML_RESTRICT x, block_e3m4_q * GGML_RESTRICT y, int64_t k);

    GGML_API void dequantize_row_fq8_4  (const block_fq8_4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void quantize_row_fq8_4_ref(const float * GGML_RESTRICT x, block_fq8_4 * GGML_RESTRICT y, int64_t k);
    GGML_API void dequantize_row_fq8_3  (const block_fq8_3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    GGML_API void quantize_row_fq8_3_ref(const float * GGML_RESTRICT x, block_fq8_3 * GGML_RESTRICT y, int64_t k);

#ifdef  __cplusplus
}
#endif
