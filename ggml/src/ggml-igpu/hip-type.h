#include "types.h"

namespace ggml::backend::igpu::type {

template<typename TIN, typename TOUT>
struct to {};

template<> struct to<float32_t, float32_t> {
    using in_t = float32_t;
    using out_t = float32_t;
    static constexpr int SIZE = 1;
    __device__ __forceinline__ void convert(in_t in, out_t& out) { out = (out_t)in; }
    __device__ __forceinline__ out_t operator()(in_t in) { return (out_t)in; }
};
template<> struct to<float16_t, float32_t> {
    using in_t = float16_t;
    using out_t = float32_t;
    static constexpr int SIZE = 1;
    __device__ __forceinline__ void convert(in_t in, out_t& out) { out = (out_t)in; }
    __device__ __forceinline__ out_t operator()(in_t in) { return (out_t)in; }
};
template<> struct to<float32_t, float16_t> {
    using in_t = float32_t;
    using out_t = float16_t;
    static constexpr int SIZE = 1;
    __device__ __forceinline__ void convert(in_t in, out_t& out) { out = (out_t)in; }
    __device__ __forceinline__ out_t operator()(in_t in) { return (out_t)in; }
};
template<> struct to<float16_t, float16_t> {
    using in_t = float16_t;
    using out_t = float16_t;
    static constexpr int SIZE = 1;
    __device__ __forceinline__ void convert(in_t in, out_t& out) { out = (out_t)in; }
    __device__ __forceinline__ out_t operator()(in_t in) { return (out_t)in; }
};
template<> struct to<bfloat16_t, float32_t> {
    using in_t = bfloat16_t;
    using out_t = float32_t;
    static constexpr int SIZE = 1;
    __device__ __forceinline__ void convert(in_t in, out_t& out) { out = (out_t)in; }
    __device__ __forceinline__ out_t operator()(in_t in) { return (out_t)in; }
};
template<> struct to<float32_t, bfloat16_t> {
    using in_t = float32_t;
    using out_t = bfloat16_t;
    static constexpr int SIZE = 1;
    __device__ __forceinline__ void convert(in_t in, out_t& out) { out = (out_t)in; }
    __device__ __forceinline__ out_t operator()(in_t in) { return (out_t)in; }
};
template<> struct to<bfloat16_t, bfloat16_t> {
    using in_t = bfloat16_t;
    using out_t = bfloat16_t;
    static constexpr int SIZE = 1;
    __device__ __forceinline__ void convert(in_t in, out_t& out) { out = (out_t)in; }
    __device__ __forceinline__ out_t operator()(in_t in) { return (out_t)in; }
};

// TODO voir pour les types quantisés...

}
