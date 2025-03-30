#include "mulmat.h"
#include "mulmat-imp.h"
#include "types.h"

namespace ggml::backend::igpu::op_mul_mat {
    // pack des poids (A): liste des template instanciables
    template bool repack<bfloat16_t, bfloat16_t>(const bfloat16_t* ref, std::size_t la, bfloat16_t* bloc, std::size_t M, std::size_t K);
    template bool repack<float16_t,  float16_t >(const float16_t*  ref, std::size_t la, float16_t*  bloc, std::size_t M, std::size_t K);
}
