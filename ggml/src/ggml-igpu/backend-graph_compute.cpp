#include "imp.h"

#include "ggml-log.h"
//#include "hip-tools.h"

#include "tools.h"
#include "types.h"
#include "ops.h"

namespace ggml::backend::igpu::imp {
enum ggml_status graph_compute(ggml_cgraph & cgraph) {
    for (int i = 0; i < cgraph.n_nodes; i++) {
        ggml_tensor * node = cgraph.nodes[i];
        auto& op = *node;
        auto& src0 = *(node->src[0]);
        auto& src1 = *(node->src[1]);
        switch (node->op) {
        case GGML_OP_UNARY:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            op::unary::compute(op, src0, op);
            // TODO voir si on peu merger une conversion:
            // ou voir le "graph_optimize"
            // op::unary::compute(op, src0, *cgraph.nodes[i+1]); // +GGML_OP_CPY
            break;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
            op::binary::compute(op, src0, src1, op);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        default:
            // __PRETTY_FUNCTION__ ?
            GGML_ABORT("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
    }
    return GGML_STATUS_SUCCESS;
}
}
