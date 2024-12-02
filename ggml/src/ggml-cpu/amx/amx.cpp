#include "amx.h"
#include "common.h"
#include "mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "traits.h"

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            return true;
        }
        return false;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}
}  // namespace ggml::cpu::amx

namespace ggml::cpu::amx {

// AMX buffer
class buffer : public ggml::cpu::buffer {
public:
    buffer(std::size_t size) : ggml::cpu::buffer(size) { }

    virtual ~buffer() { }

    ggml_status init_tensor(ggml_tensor& tensor) override {
        tensor->extra = (void *) ggml::cpu::amx::get_tensor_traits(&tensor);
        return GGML_STATUS_SUCCESS;
    }

    void set_tensor(ggml_tensor & tensor, const void * data, std::size_t offset, std::size_t size) override {
        if (qtype_has_amx_kernels(tensor.type)) {
            GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor.name, ggml_type_name(tensor.type));
            ggml_backend_amx_convert_weight(&tensor, data, offset, size);
        } else {
            memcpy((char *) tensor.data + offset, data, size);
        }
    }

    /*
    // need to figure what we need to do with buffer->extra.
    static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
        GGML_ASSERT(!qtype_has_amx_kernels(tensor->type));
        memcpy(data, (const char *)tensor->data + offset, size);

        GGML_UNUSED(buffer);
    }

    static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
        if (ggml_backend_buffer_is_host(src->buffer)) {
            if (qtype_has_amx_kernels(src->type)) {
                ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_nbytes(dst));
            } else {
                memcpy(dst->data, src->data, ggml_nbytes(src));
            }
            return true;
        }
        return false;

        GGML_UNUSED(buffer);
    }
    */

};

class extra_buffer_type : ggml::cpu::extra_buffer_type {

    const std::string& get_name() override {
        static const std::string name {"AMX"};
        return name;
    }

    ggml::cpp::backend::buffer* alloc_buffer(std::size_t size) override {
        return new buffer(size);
    }

    std::size_t get_alloc_size(const ggml_tensor& tensor) override {
        return ggml_backend_amx_get_alloc_size(&tensor);
    }

    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        // handle only 2d gemm for now
        auto is_contiguous_2d = [](const struct ggml_tensor * t) {
            return ggml_is_contiguous(t) && t->ne[3] == 1 && t->ne[2] == 1;
        };

        if (op->op == GGML_OP_MUL_MAT && is_contiguous_2d(op->src[0]) &&  // src0 must be contiguous
            is_contiguous_2d(op->src[1]) &&                               // src1 must be contiguous
            op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_amx_buffer_type() &&
            op->src[0]->ne[0] % (TILE_K * 2 * 32) == 0 && // TODO: not sure if correct (https://github.com/ggml-org/llama.cpp/pull/16315)
            op->ne[0] % (TILE_N * 2) == 0 &&                              // out_features is 32x
            (qtype_has_amx_kernels(op->src[0]->type) || (op->src[0]->type == GGML_TYPE_F16))) {
            // src1 must be host buffer
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            // src1 must be float32
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        return nullptr;
    }
};
}  // namespace ggml::cpu::amx

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

static bool ggml_amx_init() {
#if defined(__linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        return false;
    }
    return true;
#elif defined(_WIN32)
    return true;
#else
    return false;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    static auto* buffer_type = ggml::cpu::c_wrapper(new ggml::cpu::amx::extra_buffer_type());
    if (!ggml_amx_init()) {
        return nullptr;
    }
    return buffer_type;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
