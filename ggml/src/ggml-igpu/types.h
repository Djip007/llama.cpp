#pragma once
#include <cstddef>

#include "ggml.h"

namespace ggml::backend::igpu {

    // les types...
    using bfloat16_t = __bf16;
    using float16_t  = _Float16; // le type C officiel
    using float32_t  = float;

    template<ggml_type TYPE> struct to_type {};
    template<> struct to_type<GGML_TYPE_F32 >{ using type =  float32_t; };
    template<> struct to_type<GGML_TYPE_F16 >{ using type =  float16_t; };
    template<> struct to_type<GGML_TYPE_BF16>{ using type = bfloat16_t; };


}
