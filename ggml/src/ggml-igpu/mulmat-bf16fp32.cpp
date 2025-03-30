#include "mulmat.h"
#include "mulmat-imp.h"
#include "types.h"

namespace ggml::backend::igpu::op_mul_mat {
    // les instanciations:
    template bool supported<bfloat16_t, float32_t, float32_t>(const ggml_tensor& A, const ggml_tensor& B, const ggml_tensor& C);
    template bool compute<bfloat16_t, float32_t, float32_t>(const bfloat16_t* A, const float32_t* B, float32_t* C,
            std::size_t M, std::size_t N, std::size_t K, std::size_t la, std::size_t lb, std::size_t lc);

}
