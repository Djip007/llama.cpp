#pragma once
#include <cstddef>

namespace ggml::backend::igpu {

    // les types...
    using bfloat16_t = __bf16;
    using float16_t  = _Float16; // le type C officiel
    using float32_t  = float;
}
