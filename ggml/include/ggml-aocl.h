#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// backend API
// GGML_BACKEND_API bool ggml_backend_is_aocl(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_aocl_reg(void);

#ifdef __cplusplus
}
#endif
