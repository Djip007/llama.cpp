#include "ggml-igpu.h"

#include "ggml_cpp_wrapper.h"
#include "ggml-backend-impl.h"

namespace ggml::backend::igpu::imp {
bool supports_op(const ggml_tensor & op);
enum ggml_status graph_compute(ggml_cgraph & cgraph);
}
