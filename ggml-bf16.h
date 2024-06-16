#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// un POC: calcule fait avec BF16:
// - avec zen4
// - avec RDNA3 (APU seulement?)
// autre cas???

// on ne code que les matmul ?

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_bf16_init(void);

#ifdef  __cplusplus
}
#endif
