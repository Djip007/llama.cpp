#include "imp.h"

#include "ggml-log.h"
//#include "hip-tools.h"

#include "tools.h"
//#include "types.h"
#include "ops.h"

#ifdef DEV_ACTIVE
#include <iostream>
#include <unordered_set>
#endif

namespace ggml::backend::igpu::imp {
bool supports_op(const ggml_tensor & op) {
    bool supported = false;
    auto& src0 = *(op.src[0]);
    auto& src1 = *(op.src[1]);
    switch (op.op) {
    case GGML_OP_UNARY:
    case GGML_OP_SQR:
    case GGML_OP_SQRT:
    case GGML_OP_LOG:
    case GGML_OP_SIN:
    case GGML_OP_COS:
        supported = op::unary::supports(op, src0, op);
        break;
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_SUB:
    case GGML_OP_DIV:
        supported = op::binary::supports(op, src0, src1, op);
        break;
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        return true;
    // les OP pas encore implementés:
    case GGML_OP_DUP:
    case GGML_OP_ADD_ID:
    case GGML_OP_ADD1:
    case GGML_OP_ACC:
    case GGML_OP_SUM:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_MEAN:
    case GGML_OP_ARGMAX:
    case GGML_OP_COUNT_EQUAL:
    case GGML_OP_REPEAT:
    case GGML_OP_REPEAT_BACK:
    case GGML_OP_CONCAT:
    case GGML_OP_SILU_BACK:
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:  // next to do for llama-3.1
    case GGML_OP_RMS_NORM_BACK:
    case GGML_OP_GROUP_NORM:
    case GGML_OP_L2_NORM:
    case GGML_OP_MUL_MAT:
    case GGML_OP_MUL_MAT_ID:
    case GGML_OP_OUT_PROD:
    case GGML_OP_SCALE:
    case GGML_OP_SET:
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_GET_ROWS:
    case GGML_OP_GET_ROWS_BACK:
    case GGML_OP_SET_ROWS:
    case GGML_OP_DIAG:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_DIAG_MASK_ZERO:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_SOFT_MAX_BACK:
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
    case GGML_OP_CLAMP:
    case GGML_OP_CONV_TRANSPOSE_1D:
    case GGML_OP_IM2COL:
    case GGML_OP_IM2COL_BACK:
    case GGML_OP_IM2COL_3D:
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_3D:
    case GGML_OP_CONV_2D_DW:
    case GGML_OP_CONV_TRANSPOSE_2D:
    case GGML_OP_POOL_1D:
    case GGML_OP_POOL_2D:
    case GGML_OP_POOL_2D_BACK:
    case GGML_OP_UPSCALE:
    case GGML_OP_PAD:
    case GGML_OP_PAD_REFLECT_1D:
    case GGML_OP_ROLL:
    case GGML_OP_ARANGE:
    case GGML_OP_TIMESTEP_EMBEDDING:
    case GGML_OP_ARGSORT:
    case GGML_OP_LEAKY_RELU:
    case GGML_OP_FLASH_ATTN_EXT:
    case GGML_OP_FLASH_ATTN_BACK:
    case GGML_OP_SSM_CONV:
    case GGML_OP_SSM_SCAN:
    case GGML_OP_WIN_PART:
    case GGML_OP_WIN_UNPART:
    case GGML_OP_GET_REL_POS:
    case GGML_OP_ADD_REL_POS:
    case GGML_OP_RWKV_WKV6:
    case GGML_OP_GATED_LINEAR_ATTN:
    case GGML_OP_RWKV_WKV7:
    case GGML_OP_MAP_CUSTOM1:
    case GGML_OP_MAP_CUSTOM2:
    case GGML_OP_MAP_CUSTOM3:
    case GGML_OP_CUSTOM:
    case GGML_OP_CROSS_ENTROPY_LOSS:
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
    case GGML_OP_OPT_STEP_ADAMW:
    case GGML_OP_OPT_STEP_SGD:
    case GGML_OP_GLU:
    case GGML_OP_COUNT:
        supported = false;
    }
#ifdef DEV_ACTIVE
    if (!supported) {
        // histoire de lister toutes les OPs pas encores supportée... (1 seule fois)
        static std::unordered_set<std::string> list_ops;
        std::stringstream id;
        id << ggml_op_name(op.op);
        if (op.op == GGML_OP_UNARY) id << "@" << ggml_unary_op_name(ggml_get_unary_op(&op));
        id <<"#"<<ggml_type_name(op.type);
        id <<"#"<<op.ne[0];
        id <<"#"<<op.ne[1];
        id <<"#"<<op.ne[2];
        id <<"#"<<op.ne[3];
        for (int i=0; i<GGML_MAX_SRC; ++i) {
            if (op.src[i] != nullptr) {
                id <<"#"<<ggml_type_name(op.src[i]->type);
                id <<"#"<<op.src[i]->ne[0];
                id <<"#"<<op.src[i]->ne[1];
                id <<"#"<<op.src[i]->ne[2];
                id <<"#"<<op.src[i]->ne[3];
            }
        }
        if (list_ops.count(id.str()) == 0) {
            list_ops.insert(id.str());
            IGPU_DEV("##>> op("<< op.name<<"<"<<ggml_op_name(op.op)<<((op.op == GGML_OP_UNARY)?"@":"")<<((op.op == GGML_OP_UNARY)?ggml_unary_op_name(ggml_get_unary_op(&op)):"")<<">) : "
                    << ggml_type_name(op.type)
                    << "[" <<op.ne[0]<<", "<<op.ne[1]<<", "<<op.ne[2]<<", "<<op.ne[3]<<"]");
            for (int i=0; i<GGML_MAX_SRC; ++i) {
                if (op.src[i] != nullptr) {
                    IGPU_DEV("   {"<<i<<"} " << op.src[i]->name << "<" <<ggml_type_name(op.src[i]->type)<<"> "
                            << "["<<op.src[i]->ne[0]<<", "<<op.src[i]->ne[1]<<", "<<op.src[i]->ne[2]<<", "<<op.src[i]->ne[3]<<"]");
                }
            }
        }
    }
#endif
    return supported;
}
}