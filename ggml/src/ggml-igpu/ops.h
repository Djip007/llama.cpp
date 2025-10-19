// la liste des OP
#include "ggml.h"

namespace ggml::backend::igpu::op {

// unity ops and similare...
struct unary {
    static bool supports(const ggml_tensor & op, const ggml_tensor & in, const ggml_tensor & out);
    static void compute(const ggml_tensor & op, const ggml_tensor & in, ggml_tensor & out);
};

// binary like OP (element wide with broadcast)
struct binary {
    static bool supports(const ggml_tensor & op, const ggml_tensor & in0, const ggml_tensor & in1, const ggml_tensor & out);
    static void compute(const ggml_tensor & op, const ggml_tensor & in0, const ggml_tensor & in1, ggml_tensor & out);
};

//

}
