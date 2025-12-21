#include "ggml.h"
#include "ggml-log.h"

#include "ggml-aocl.h"
#include "tools.h"

#include "ggml_cpp_wrapper.h"
#include "ggml-backend-impl.h"

// pour les calcules ...
#include <aocl_dlp.h>
#include <sstream>
#include <vector>
#include <assert.h>
#include <unistd.h>
#include <immintrin.h>

#include "aocl.hpp"

/*
This backend use amd aocl-dlp lib.
 - gemm
 - batched_gemm
 - repacked

For type:
  - fp32
  - bf16

for use with zen4/zen5/... CPU
*/

namespace ggml::backend::aocl {

    using bfloat16_t = ::aocl::bfloat16_t;
    using float32_t  = ::aocl::float32_t;

    enum class BUFFER_TYPE {
        HOST,
        //+ EXTRA buffers for tensor repacking
        REPACK_FP32xFP32xFP32,
        REPACK_BF16xFB16xFP32,
    };

    constexpr const char* to_char(BUFFER_TYPE t) {
        switch (t) {
        case BUFFER_TYPE::HOST:                   return "host";
        case BUFFER_TYPE::REPACK_FP32xFP32xFP32:  return "repack_fp32fp32fp32";
        case BUFFER_TYPE::REPACK_BF16xFB16xFP32:  return "repack_bf16bf16fp32";
        }
    }

    static int NEXT_BUFFER_ID = 0;

    class buffer : public ggml::cpp::backend::buffer {
    protected:
        uint8_t* m_data = nullptr;
        const std::size_t m_size;
        const int m_id;
        const BUFFER_TYPE m_type;
    public:
        buffer(BUFFER_TYPE type, std::size_t size) : m_size(size), m_id(NEXT_BUFFER_ID++), m_type(type) {
            m_data = new (std::align_val_t(::aocl::GET_TENSOR_ALIGNMENT())) uint8_t[m_size];
        }
        virtual ~buffer() {
            delete[] m_data;
        }
        void* get_base() override {
            return m_data;
        }
    };

    class host_buffer : public buffer {
    public:
        host_buffer(std::size_t size) : buffer(BUFFER_TYPE::HOST, size) { }
        void memset_tensor(ggml_tensor & tensor, uint8_t value, std::size_t offset, std::size_t size) override {
            GGML_ASSERT(value == 0); // pas de sens sinon...
            memset((uint8_t *) tensor.data + offset, value, size);
        }
        void set_tensor(ggml_tensor & tensor, const void * data, std::size_t offset, std::size_t size) override {
            memcpy((uint8_t *)tensor.data + offset, data, size);
        }
        void get_tensor(const ggml_tensor & tensor, void * data, std::size_t offset, std::size_t size) override {
            memcpy(data, (uint8_t *)tensor.data + offset, size);
        }
        void clear(uint8_t value) override {
            GGML_ASSERT(value == 0); // pas de sens sinon...
            memset(m_data, value, m_size);
        }
    };

    // buffer for tensor repacking
    class repack_buffer : public buffer {
    public:
        repack_buffer(BUFFER_TYPE type, std::size_t size) : buffer(type, size) { }
        ggml_status init_tensor(ggml_tensor & tensor) override {
            // Permet si besoin de definir qq-chose sur extra. => pour repacking!
            GGML_ASSERT(tensor.extra == nullptr);
            return GGML_STATUS_SUCCESS;
        }
        void memset_tensor(ggml_tensor & tensor, uint8_t , std::size_t , std::size_t ) override {
            GGML_ASSERT(tensor.data == nullptr); // Non ne doit pas etre utilisé
        }
        void get_tensor(const ggml_tensor & tensor, void * , std::size_t , std::size_t ) override {
            GGML_ASSERT(tensor.data == nullptr); // Non ne doit pas etre utilisé
        }
        void clear(uint8_t value) override {
            GGML_ASSERT(value != value); // Non ne doit pas etre utilisé
        }
    };
    class repack_buffer_fp32xfp32xfp32 : public repack_buffer {
    public:
        repack_buffer_fp32xfp32xfp32(std::size_t size) : repack_buffer(BUFFER_TYPE::REPACK_FP32xFP32xFP32, size) { }
        void set_tensor(ggml_tensor & tensor, const void * data, std::size_t offset, std::size_t size) override {
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(ggml_nbytes(&tensor) == size);
            GGML_ASSERT(tensor.nb[0] == 4); // sizeof(FP32)!
            // repacking...
            const auto K = tensor.ne[0];
            const auto M = tensor.ne[1];
            const auto ldin  = tensor.nb[1]/4;
            const auto ldout = ::aocl::get_reorder_size_fp32fp32fp32(M, K);

            for (int64_t i3 = 0; i3<tensor.ne[3]; ++i3)
            for (int64_t i2 = 0; i2<tensor.ne[2]; ++i2) {
                auto in  = (const float*) ((const uint8_t*)data        + (i2*tensor.nb[2]+i3*tensor.nb[3]));
                auto out = (      float*) ((      uint8_t*)tensor.data + (i2+i3*tensor.ne[2])*ldout);
                aocl_reorder_f32f32f32of32('R', 'T', 'B', in, out, K, M, ldin, nullptr);
            }
        }
    };
    class repack_buffer_bf16xbf16xfp32 : public repack_buffer {
    public:
        repack_buffer_bf16xbf16xfp32(std::size_t size) : repack_buffer(BUFFER_TYPE::REPACK_BF16xFB16xFP32, size) { }
        void set_tensor(ggml_tensor & tensor, const void * data, std::size_t offset, std::size_t size) override {
            GGML_ASSERT(offset == 0);
            GGML_ASSERT(ggml_nbytes(&tensor) == size);
            GGML_ASSERT(tensor.nb[0] == 2); // BF16
            // repacking...
            const auto K = tensor.ne[0];
            const auto M = tensor.ne[1];
            const auto ldin  = tensor.nb[1]/2;
            const auto ldout = ::aocl::get_reorder_size_bf16bf16fp32(M, K);

            for (int64_t i3 = 0; i3<tensor.ne[3]; ++i3)
            for (int64_t i2 = 0; i2<tensor.ne[2]; ++i2) {
                auto in  = (const bfloat16*) ((const uint8_t*)data        + (i2*tensor.nb[2]+i3*tensor.nb[3]));
                auto out = (      bfloat16*) ((      uint8_t*)tensor.data + (i2+i3*tensor.ne[2])*ldout);
                aocl_reorder_bf16bf16f32of32('R', 'T', 'B', in, out, K, M, ldin, nullptr);
            }
        }
    };

    // the buffers types:
    //  - host general case
    //  - extras for repacking
    class buffer_type : public ggml::cpp::backend::buffer_type {
        const int m_deviceId;
        const std::string m_name;
        const BUFFER_TYPE m_type;

        static const std::string name_format(int id, BUFFER_TYPE type) {
            std::ostringstream ostr;
            switch (type) {
            case BUFFER_TYPE::HOST:
                ostr << "AOCL_host_buffer<"<<id<<">"; // pas forcement utils
                break;
            case BUFFER_TYPE::REPACK_FP32xFP32xFP32:
                ostr << "AOCL_repack_buffer_fp32fp32fp32<"<<id<<">";
                break;
            case BUFFER_TYPE::REPACK_BF16xFB16xFP32:
                ostr << "AOCL_repack_buffer_bf16bf16fp32<"<<id<<">";
                break;
            //case case BUFFER_TYPE::EXTRA_ZZZ:
            //    ostr << "AOCL_extra_zzz_buffer<"<<id<<">";
            //    break;
            default :
                ostr << "AOCL_???_buffer<"<<id<<">";
            }
            return ostr.str();
        }

    public:
        buffer_type(int devideId, BUFFER_TYPE type): m_deviceId(devideId), m_name(name_format(m_deviceId, type)), m_type(type) { }
        virtual ~buffer_type() { }
        const std::string& get_name() override {
            return m_name;
        }
        ggml::cpp::backend::buffer* alloc_buffer(std::size_t size) override {
            switch (m_type) {
            case BUFFER_TYPE::HOST:
                return new host_buffer(size);
            case BUFFER_TYPE::REPACK_FP32xFP32xFP32:
                return new repack_buffer_fp32xfp32xfp32(size);
            case BUFFER_TYPE::REPACK_BF16xFB16xFP32:
                return new repack_buffer_bf16xbf16xfp32(size);
            //case BUFFER_TYPE::EXTRA_ZZZ:
            //    retrun new ...;
            default :
                return nullptr;
            }
        }

        std::size_t get_alignment() override { return ::aocl::GET_TENSOR_ALIGNMENT(); }
        std::size_t get_max_size() override { return SIZE_MAX; }
        std::size_t get_alloc_size(const ggml_tensor& tensor) override {
            switch (m_type) {
            case BUFFER_TYPE::HOST: {
                return ggml_nbytes(&tensor);
            }
            case BUFFER_TYPE::REPACK_FP32xFP32xFP32: {
                return ::aocl::get_reorder_size_fp32fp32fp32(tensor.ne[1], tensor.ne[0])*tensor.ne[2]*tensor.ne[3];
            }
            case BUFFER_TYPE::REPACK_BF16xFB16xFP32: {
                return ::aocl::get_reorder_size_bf16bf16fp32(tensor.ne[1], tensor.ne[0])*tensor.ne[2]*tensor.ne[3];
            }
            //case BUFFER_TYPE::EXTRA_ZZZ:
            //    retrun <size>;
            default :
                return 0;
            }
        }
        bool is_host() override {
            return m_type == BUFFER_TYPE::HOST;
        }
    };

    class backend;

    class device : public ggml::cpp::backend::device {
        const std::string m_name;
        const std::string m_desc;
        const int m_id;

        buffer_type* m_host_buffer_type;
        buffer_type* m_repack_fp32fp32fp32_buffer_type;
        buffer_type* m_repack_bf16fb16fp32_buffer_type;
        // buffer_type* m_extra_zzzz_buffer_type;

    public:
        device(const std::string& name, int deviceId, const std::string& desc = "...") : m_name(name), m_desc(desc), m_id(deviceId) {
            GGML_LOG_INFO("ggml-aocl: device[%s::%d] added: %s\n", m_name.c_str(), m_id, m_desc.c_str());
            // Lié au device puisse que c'est lui qui alloue les buffer sur/pour le bon device!
            m_host_buffer_type   = new buffer_type(m_id, BUFFER_TYPE::HOST);
            m_repack_fp32fp32fp32_buffer_type = new buffer_type(m_id, BUFFER_TYPE::REPACK_FP32xFP32xFP32);
            m_repack_bf16fb16fp32_buffer_type = new buffer_type(m_id, BUFFER_TYPE::REPACK_BF16xFB16xFP32);
            // publication des extra_buffer.
            register_extra_buffer_type(m_repack_bf16fb16fp32_buffer_type);
            register_extra_buffer_type(m_repack_fp32fp32fp32_buffer_type);
        }
        virtual ~device() {
            // TODO: il faudrait aussi detruire les wrappers?
            delete m_host_buffer_type;     m_host_buffer_type=nullptr;
            delete m_repack_fp32fp32fp32_buffer_type;   m_repack_fp32fp32fp32_buffer_type=nullptr;
            delete m_repack_bf16fb16fp32_buffer_type;   m_repack_bf16fb16fp32_buffer_type=nullptr;
            // delete m_extra_zzzz_buffer_type; m_extra_zzzz_buffer_type=nullptr;
        }
        const std::string& get_name() override {
            return m_name;
        }
        const std::string& get_description() override {
            return m_desc;
        }
        void get_memory(std::size_t & free, std::size_t & total) override {
            long pages = sysconf(_SC_PHYS_PAGES);
            long page_size = sysconf(_SC_PAGE_SIZE);
            total = pages * page_size;
            // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
            free = total;
        }
        enum ggml_backend_dev_type get_type() override {
            //return GGML_BACKEND_DEVICE_TYPE_CPU;
            //return GGML_BACKEND_DEVICE_TYPE_ACCEL;
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        }
        ggml::cpp::backend::backend& init_backend(const std::string& params) override;
        ggml::cpp::backend::buffer_type& get_buffer_type() override {
            return *m_host_buffer_type;
        }

        // Pas sur que ca soit util... mais bon
        bool caps_host_buffer() override { return true; }
        ggml::cpp::backend::buffer_type* get_host_buffer_type() override {
            return m_host_buffer_type;
        }

        bool supports_op(const ggml_tensor & op) override {
            // avec les extra ca peu dependre du type des buffer?
            bool supported = false;
            auto& src0 = *(op.src[0]);
            auto& src1 = *(op.src[1]);
            switch (op.op) {
                case GGML_OP_NONE:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                    supported = true;
                    break;
                case GGML_OP_MUL_MAT: {
                    if ((src0.nb[0] == ggml_type_size(src0.type))
                     && (src1.nb[0] == ggml_type_size(src1.type))
                     && (  op.nb[0] == ggml_type_size(  op.type))
                    ){
                        if (src0.type == GGML_TYPE_F32  && src1.type == GGML_TYPE_F32  && op.type == GGML_TYPE_F32 ) {
                            supported = (src0.buffer == nullptr)
                                     || (src0.buffer->buft->context == (void*) m_repack_fp32fp32fp32_buffer_type)
                                     || (src0.buffer->buft->context == (void*) m_host_buffer_type);
                        }
                        if (src0.type == GGML_TYPE_BF16 && src1.type == GGML_TYPE_BF16 && op.type == GGML_TYPE_F32 ) {
                            supported = (src0.buffer == nullptr)
                                     || (src0.buffer->buft->context == (void*) m_repack_bf16fb16fp32_buffer_type)
                                     || (src0.buffer->buft->context == (void*) m_host_buffer_type);
                        }
                        if (src0.type == GGML_TYPE_BF16 && src1.type == GGML_TYPE_F32  && op.type == GGML_TYPE_F32 ) {
                            // ?? dans quel cas on a un nullptr???
                            supported = (src0.buffer == nullptr)
                                     || (src0.buffer->buft->context == (void*) m_repack_bf16fb16fp32_buffer_type)
                                     || (src0.buffer->buft->context == (void*) m_host_buffer_type);
                        }
                    }
                    if (!supported) {
                        AOCL_TRACE(ggml_type_name(op.type) << "=" << ggml_type_name(src0.type) << "@" << ggml_type_name(src1.type)
                                   << ": " << src0.ne[1] << "/" << src1.ne[1] << "/" << src0.ne[0]
                                   << " => " << op.ne[2] << "/" << op.ne[3]
                                   << ":" << src0.ne[2] <<"/"<< src0.ne[3] 
                                   << ":" << src1.ne[2] <<"/"<< src1.ne[3] 
                                   );
                    }
                } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    // AOCL_TRACE("TODO: supported GGML_OP_FLASH_ATTN_EXT");
                default:
                    supported = false;
            }

            return supported;
        }
        bool supports_buft(ggml_backend_buffer_type_t buffer_type) override {
            // - Cas CPU: tout ceux qui sont en RAM 
            if (ggml_backend_buft_is_host(buffer_type)) {
                return true;
            }
            // - cas AOCL extra: limité au siens pour l'instant, les autres sont à configurer/copier.
            if (buffer_type->context) {
                // + nos propres buffers
                if (buffer_type->context == (void*) m_host_buffer_type)                return true;
                if (buffer_type->context == (void*) m_repack_fp32fp32fp32_buffer_type) return true;
                if (buffer_type->context == (void*) m_repack_bf16fb16fp32_buffer_type) return true;
                // if (buffer_type->context == (void*) m_extra_zzzz_buffer_type) return true;
            }
            return false;
        }
        // les methodes locales:
        int getID() const {return m_id;}
        bool is_repacked(const ggml_tensor & tensor) {
            return (tensor.buffer->buft->context == (void*) m_repack_fp32fp32fp32_buffer_type)
                 ||(tensor.buffer->buft->context == (void*) m_repack_bf16fb16fp32_buffer_type);
        }
    };

    class backend : public ggml::cpp::backend::backend {
        const int m_deviceId;
        int m_nb_threads = 1;

        // de quoi convertir les tenseurs
        void* m_tmp_buffer = nullptr;
        std::size_t m_tmp_size = 0;
        template<typename T>
        T* get_tmp(std::size_t size) {
            std::size_t nb_byte = size * sizeof(T);
            if (nb_byte > m_tmp_size) {
                nb_byte = std::max(nb_byte , 2*m_tmp_size);
                // force "aligned size"
                nb_byte = ((nb_byte-1)/::aocl::GET_TENSOR_ALIGNMENT())+1;
                nb_byte *= ::aocl::GET_TENSOR_ALIGNMENT();
                if (m_tmp_buffer) std::free(m_tmp_buffer);
                m_tmp_size = nb_byte;
                m_tmp_buffer = aligned_alloc(::aocl::GET_TENSOR_ALIGNMENT(), m_tmp_size);
            }
            return (T*) m_tmp_buffer;
        }

    public:
        backend(const std::string& params, device& dev) :
            ggml::cpp::backend::backend(dev), m_deviceId(dev.getID())
        {}
        virtual ~backend() {
            if (m_tmp_buffer) std::free(m_tmp_buffer);
        }
        const std::string& get_name() override {
            return m_device.get_name();
        }
        ggml_guid_t get_guid() override {
            // uuidgen >  0a15687b-8921-4898-af8c-aca8abb17308
            static ggml_guid guid = { 0x0a, 0x15, 0x68, 0x7b, 0x89, 0x21, 0x48, 0x98, 0xaf, 0x8c, 0xac, 0xa8, 0xab, 0xb1, 0x73, 0x08 };
            return &guid;
        }
        enum ggml_status graph_compute(ggml_cgraph & cgraph) override {
            device& dev = static_cast<device&>(m_device);
            for (int i = 0; i < cgraph.n_nodes; i++) {
                ggml_tensor * node = cgraph.nodes[i];
                auto& op = *node;
                switch (op.op) {
                    case GGML_OP_NONE:
                    case GGML_OP_RESHAPE:
                    case GGML_OP_VIEW:
                    case GGML_OP_PERMUTE:
                    case GGML_OP_TRANSPOSE:
                        break;
                    case GGML_OP_MUL_MAT: if (op.src[1]->ne[1] > 0) {
                        auto& A = *(op.src[0]);  // poids: ici repacking possible
                        auto& B = *(op.src[1]);  // activation: ici changement de type possible
                        auto& C = op;
                        GGML_ASSERT(A.ne[0] == B.ne[0]); // K
                        GGML_ASSERT(B.ne[1] == C.ne[1]); // N
                        GGML_ASSERT(A.ne[1] == C.ne[0]); // M

                        const auto* A_data = A.data;
                        const auto* B_data = B.data;
                              auto* C_data = C.data;
                        auto A_type = A.type;
                        auto B_type = B.type;
                        auto C_type = C.type;

                        auto B_NE1 = B.ne[1];
                        auto B_NE2 = B.ne[2];
                        auto B_NE3 = B.ne[3];
                        auto B_NB0 = B.nb[0];
                        auto B_NB1 = B.nb[1];
                        auto B_NB2 = B.nb[2];
                        auto B_NB3 = B.nb[3];

                        auto C_NE1 = C.ne[1];
                        auto C_NE2 = C.ne[2];
                        auto C_NE3 = C.ne[3];
                        auto C_NB0 = C.nb[0];
                        auto C_NB1 = C.nb[1];
                        auto C_NB2 = C.nb[2];
                        auto C_NB3 = C.nb[3];

                        // conversion de B si necessaire:
                        if (A.type == GGML_TYPE_BF16 && B.type == GGML_TYPE_F32 && C.type == GGML_TYPE_F32) {
                            // il faut convertir src1 en BF16! et l'alligner correctement.
                            std::size_t ldb1 = ((B.ne[0]-1)/::aocl::GET_TENSOR_ALIGNMENT()+1)*::aocl::GET_TENSOR_ALIGNMENT();
                            std::size_t ldb2 = ldb1*B_NE1;
                            std::size_t ldb3 = ldb2*B_NE2;
                            std::size_t sizeB = ldb1 * B_NE1 * B_NE2 * B_NE3;
                            auto* tmp = get_tmp<bfloat16_t>(std::max(sizeB, B.ne[0]*(std::size_t)1024));
#                           pragma omp parallel for collapse(3) num_threads(m_nb_threads) schedule(static)
                            for (int64_t i3 = 0; i3<B_NE3; ++i3)
                            for (int64_t i2 = 0; i2<B_NE2; ++i2)
                            for (int64_t i1 = 0; i1<B_NE1; ++i1) {
                                ::aocl::convert((float*)((uint8_t*)B.data+i1*B_NB1+i2*B_NB2+i3*B_NB3), tmp+ldb1*i1+ldb2*i2+ldb3*i3, B.ne[0]);
                            }
                            // reconfig B:
                            B_data = tmp;
                            B_type = GGML_TYPE_BF16;
                            B_NB0 = 2;
                            B_NB1 = ldb1*B_NB0;
                            B_NB2 = ldb2*B_NB0;
                            B_NB3 = ldb3*B_NB0;
                        }

                        // est-ce que les N2 se "suivent""
                        if (B_NB2 == B_NB1*B_NE1 && C.nb[2] == C.nb[1]*C_NE1) {
                            const int64_t r2 = B.ne[2]/A.ne[2];
                            B_NE1 = B_NE1*r2;
                            B_NE2 = B_NE2/r2;
                            B_NB2 = B_NB2*r2;
                            C_NE1 = C_NE1*r2;
                            C_NE2 = C_NE2/r2;
                            C_NB2 = C_NB2*r2;
                        }

                        bool repack = dev.is_repacked(A);
                        const auto M = A.ne[1]; // == C.ne[0]
                        const auto N = B_NE1;   // == C.ne[1]*r2 ?
                        const auto K = A.ne[0]; // == B.ne[0]
                        AOCL_TRACE("N? :" << N <<"/"<< C.ne[1] <<"/"<< B.ne[1]);
                        std::size_t lda1 = A.nb[1]/A.nb[0];
                        std::size_t ldb1 = B_NB1/B_NB0;
                        std::size_t ldc1 = C_NB1/C.nb[0];
                        std::size_t lda2 = A.nb[2]/A.nb[0];
                        std::size_t ldb2 = B_NB2/B_NB0;
                        std::size_t ldc2 = C_NB2/C.nb[0];
                        std::size_t lda3 = A.nb[3]/A.nb[0];
                        std::size_t ldb3 = B_NB3/B_NB0;
                        std::size_t ldc3 = C_NB3/C.nb[0];

                        // broadcast factors
                        const int64_t r2 = B_NE2/A.ne[2];
                        const int64_t r3 = B_NE3/A.ne[3];
                        
                        const float32_t*  A_fp32 = (const float32_t*)  A_data;
                        const bfloat16_t* A_bf16 = (const bfloat16_t*) A_data;
                        const float32_t*  B_fp32 = (const float32_t*)  B_data;
                        const bfloat16_t* B_bf16 = (const bfloat16_t*) B_data;
                              float32_t*  C_fp32 = (      float32_t*)  C_data;
                        if (B_NE2*B_NE3 == 1) {
                            if (A_type == GGML_TYPE_F32 && B_type == GGML_TYPE_F32 && C_type == GGML_TYPE_F32) {
                                if (repack) {
                                    ::aocl::mul_mat<true>(A_fp32, B_fp32, C_fp32, M, N, K, lda1, ldb1, ldc1);
                                } else {
                                    ::aocl::mul_mat<false>(A_fp32, B_fp32, C_fp32, M, N, K, lda1, ldb1, ldc1);
                                }
                            } else
                            if (A_type == GGML_TYPE_BF16 && B_type == GGML_TYPE_BF16 && C_type == GGML_TYPE_F32) {
                                if (repack) {
                                    ::aocl::mul_mat<true>(A_bf16, B_bf16, C_fp32, M, N, K, lda1, ldb1, ldc1);
                                } else {
                                    ::aocl::mul_mat<false>(A_bf16, B_bf16, C_fp32, M, N, K, lda1, ldb1, ldc1);
                                }
                            } else {
                                return GGML_STATUS_FAILED;
                            }
                        } else {
                            // use batched gemm
                            assert(C_NE2==B_NE2);
                            assert(C_NE3==B_NE3);
                            std::vector<float32_t*> C_fp32_v (C_NE2*C_NE3, nullptr);
                            if (A_type == GGML_TYPE_F32 && B_type == GGML_TYPE_F32 && C_type == GGML_TYPE_F32) {
                                std::vector<const float32_t*> A_fp32_v (C_NE2*C_NE3, nullptr);
                                std::vector<const float32_t*> B_fp32_v (C_NE2*C_NE3, nullptr);
                                for (int64_t j3 = 0; j3 < B_NE3; ++j3) {
                                    for (int64_t j2 = 0; j2 < B_NE2; ++j2) {
                                        auto lda = (j2/r2)*lda2+(j3/r3)*lda3;
                                        auto ldb = j2*ldb2+j3*ldb3;
                                        auto ldc = j2*ldc2+j3*ldc3;
                                        A_fp32_v[j2+j3*B_NE2] = A_fp32+lda;
                                        B_fp32_v[j2+j3*B_NE2] = B_fp32+ldb;
                                        C_fp32_v[j2+j3*B_NE2] = C_fp32+ldc;
                                    }
                                }
                                if (repack) {
                                    ::aocl::mul_mat_batch<true>(A_fp32_v, B_fp32_v, C_fp32_v, M, N, K, lda1, ldb1, ldc1);
                                } else {
                                    ::aocl::mul_mat_batch<false>(A_fp32_v, B_fp32_v, C_fp32_v, M, N, K, lda1, ldb1, ldc1);
                                }
                            } else
                            if (A_type == GGML_TYPE_BF16 && B_type == GGML_TYPE_BF16 && C_type == GGML_TYPE_F32) {
                                std::vector<const bfloat16_t*> A_bf16_v (C_NE2*C_NE3, nullptr);
                                std::vector<const bfloat16_t*> B_bf16_v (C_NE2*C_NE3, nullptr);
                                for (int64_t j3 = 0; j3 < B_NE3; ++j3) {
                                    for (int64_t j2 = 0; j2 < B_NE2; ++j2) {
                                        auto lda = (j2/r2)*lda2+(j3/r3)*lda3;
                                        auto ldb = j2*ldb2+j3*ldb3;
                                        auto ldc = j2*ldc2+j3*ldc3;
                                        A_bf16_v[j2+j3*B_NE2] = A_bf16+lda;
                                        B_bf16_v[j2+j3*B_NE2] = B_bf16+ldb;
                                        C_fp32_v[j2+j3*B_NE2] = C_fp32+ldc;
                                    }
                                }
                                if (repack) {
                                    ::aocl::mul_mat_batch<true>(A_bf16_v, B_bf16_v, C_fp32_v, M, N, K, lda1, ldb1, ldc1);
                                } else {
                                    ::aocl::mul_mat_batch<false>(A_bf16_v, B_bf16_v, C_fp32_v, M, N, K, lda1, ldb1, ldc1);
                                }
                            }
                        }
                    } break;
                    default:
                        return GGML_STATUS_FAILED;
                }
            }
            return GGML_STATUS_SUCCESS;
        }
        void set_n_threads(int n_threads) override {
            m_nb_threads = n_threads;
            // @VOIR... 2eme cas
            //void dlp_thread_set_num_threads(md_t n_threads)
            //void dlp_thread_set_ways(md_t jc, md_t ic)
            dlp_thread_set_num_threads(n_threads);
        }
    };

    ggml::cpp::backend::backend& device::init_backend(const std::string& params) {
        // recuperer pour gerer un device, et detruit par l'appelant.
        //  c'est plus un new que un init.  => @ voir si on n'en fait pas un pointer!
        auto back = new backend(params, *this);
        return *back;
    }

    class reg: public ggml::cpp::backend::reg {
        const std::string m_name{"AOCL"};
        device* m_device;
    public:
        reg() {
            GGML_LOG_INFO("ggml-aocl: backend[%s] create\n", m_name.c_str());
            // voir comment recupere qq-chose pour la description...
            //  et p'etre eviter de le crer si on ne support pas les OPs...
            // if (supported)
            m_device = new device(m_name, 0, "???"/*desc.str()*/);
        }
        virtual ~reg() {
        }
        const std::string& get_name() override {
            return m_name;
        }
        std::size_t get_device_count() override {
            return m_device?1:0;
        }
        device& get_device(std::size_t index) override {
            GGML_ASSERT(index == 0);
            return *m_device;
        }
    };

}

ggml_backend_reg_t ggml_backend_aocl_reg(void) {
    static ggml::backend::aocl::reg ctx;
    // si il y a des devices on retourne le backend.
    if (ctx.get_device_count() > 0) return ggml::cpp::backend::c_wrapper(&ctx);
    // pas d'AOCL => pas la peine de finaliser le backend.
    return nullptr;
}

GGML_BACKEND_DL_IMPL(ggml_backend_aocl_reg)
