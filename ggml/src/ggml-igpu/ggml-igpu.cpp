#include "ggml-impl.h"
#include "ggml-igpu.h"
#include "ggml-backend-impl.h"

#include "ggml_cpp_wrapper.h"

#include "ggml-log.h"

#include "ggml-hip.h"
#include "tensor.h"

#include <iostream>
#include <vector>
#include <unordered_set>

#include "mulmat.h"
#include "tools.h"

/*
#> version bloc-bf16 V0.
  => repack de A<512,16,16>: A[K/512][M/16][K1=512/K0][M0=16][K0=16]
  => B et C sont repacké dynamiquement par le kernel_hip.

@ voir:
> cas bf16: meme repacking pour CPU et GPU ???

- A[K/(16*16|32)=256|512][M/16][k1=8*16/8*32][m0=16][k0=2]
- B devra etre re-packé d'une facon similaire
  -pp: ATTENTION N n'est pas un %16!
  -tg: faire 1 cas specifiques (pure CPU pour commancer sur 1/2 des coeurs? / GPU avec v_dual_dot2acc_f32_bf16)

> cas fp8:

 */

namespace ggml::backend::igpu {

    static bool IS_WEIGHT = true;  // TODO plutot is repacked / type de repack ?
    static bool IS_OTHER = true;
    enum class BUFFER_TYPE {
        HOST,
        DEVICE,
        EXTRA
    };

    constexpr const char* to_char(BUFFER_TYPE t) {
        switch (t) {
        case BUFFER_TYPE::HOST:   return "host";
        case BUFFER_TYPE::DEVICE: return "device";
        case BUFFER_TYPE::EXTRA:  return "extra";
        }
    }

    // pour l'instant un simple buffer en RAM => @ gerer avec hip!
    static int NEXT_BUFFER_ID = 0;
    class buffer : public ggml::cpp::backend::buffer {
        const BUFFER_TYPE m_type;
        const int m_id;

    public:
        buffer(std::size_t size, BUFFER_TYPE type) : m_type(type), m_id(NEXT_BUFFER_ID++), m_size(size) {
            IGPU_TRACE("buffer[IGPU<"<<m_id<<">]::new(" << size << ", " << to_char(m_type) << ")");
            // - cas RAM/CPU
            //m_data = new (std::align_val_t(32)) uint8_t[m_size];
            //GGML_ASSERT(m_data);
            // - cas HIP/IGPU
            m_host_data = ggml::hip::allocateHost<uint8_t>(m_size);
            m_device_data = ggml::hip::getDeviceMem(m_host_data);
        }
        virtual ~buffer() {
            IGPU_TRACE("buffer[IGPU<"<<m_id<<">]: deleted");
            // - cas CPU
            // delete [] m_data;
            // - cas IGPU
            ggml::hip::deallocateHost(m_host_data);
        }
        void* get_base() override {
            return m_host_data;
        }
        ggml_status init_tensor(ggml_tensor & tensor) override {
            const auto K = tensor.ne[0];
            const auto M = tensor.ne[1];
            //  tensor->buffer.usage
            IGPU_TRACE("init_tensor[IGPU<"<<m_id<<":"<<tensor.buffer->usage<<">]<" << tensor.name << ">[" << K << "," << M<< "]: " << to_char(m_type));

            // Permet si besoin de definir qq-chose sur extra si besoin/util.
            // ?? est-ce que tensor.data est deja valué??? => si oui on peu re-mapper dans extra / host/device
            GGML_ASSERT(tensor.extra == nullptr);
            // IS_OTHER
            if (m_type == BUFFER_TYPE::EXTRA) {
                tensor.extra = &IS_WEIGHT;
                // TODO: il faudrait savoir si il faut le reformaté
                // GGML_LOG_INFO("ggml-igpu: weight tensor<%s[%zu,%zu]> added\n", tensor.name, (std::size_t)K, (std::size_t)M);
            } else {
                tensor.extra = &IS_OTHER;
            }
            return GGML_STATUS_SUCCESS;
        }
        void memset_tensor(ggml_tensor & tensor, uint8_t value, std::size_t offset, std::size_t size) override {
            GGML_ASSERT(value == 0); // pas de sens sinon...
            memset((uint8_t *) tensor.data + offset, value, size);
        }
        void set_tensor(ggml_tensor & tensor, const void * data, std::size_t offset, std::size_t size) override {
            const auto K = tensor.ne[0];
            const auto M = tensor.ne[1];
            IGPU_TRACE("set_tensor[IGPU<"<<m_id<<":"<<tensor.buffer->usage<<">]<" << tensor.name << ">[" << K << "," << M<< "]: " << to_char(m_type));
            // - si non reformaté:
            if (tensor.extra == &IS_OTHER) {
                // pas un poids...
                memcpy((uint8_t *)tensor.data + offset, data, size);
                return;
            }
            /*
            // TODO reformater?
            if (tensor.type == GGML_TYPE_F32) {
                GGML_LOG_INFO("ggml-igpu: weight tensor<%s[%zu,%zu]> FP32\n", tensor.name, (std::size_t)K, (std::size_t)M);
                memcpy((uint8_t *)tensor.data + offset, data, size);
                return;
            }
             */
            // - Version re-formaté:
            const auto la = tensor.nb[1]/tensor.nb[0];
            if (tensor.type == GGML_TYPE_BF16) {
                // GGML_LOG_INFO("ggml-igpu: weight tensor<%s[%zu,%zu]> BF16 => reformat!\n", tensor.name, (std::size_t)K, (std::size_t)M);
                bfloat16_t* ref = (bfloat16_t*)data;
                bfloat16_t* bloc = (bfloat16_t*)tensor.data;
                op_mul_mat::repack(ref, la, bloc, M, K);
                return;
            }
            if (tensor.type == GGML_TYPE_F16) {
                // GGML_LOG_INFO("ggml-igpu: weight tensor<%s[%zu,%zu]> FP16 => reformat!\n", tensor.name, (std::size_t)K, (std::size_t)M);
                float16_t* ref = (float16_t*)data;
                float16_t* bloc = (float16_t*)tensor.data;
                op_mul_mat::repack(ref, la, bloc, M, K);
                return;
            }
            // pour l'instant c'est tout:
            GGML_ASSERT(false);
        }
        void get_tensor(const ggml_tensor & tensor, void * data, std::size_t offset, std::size_t size) override {
            const auto K = tensor.ne[0];
            const auto M = tensor.ne[1];
            IGPU_TRACE("get_tensor[IGPU<"<<m_id<<">]<" << tensor.name << ">[" << K << "," << M<< "]: " << to_char(m_type));
            // OK si non reformaté
            memcpy(data, (uint8_t *)tensor.data + offset, size);
        }
        void clear(uint8_t value) override {
            GGML_ASSERT(value == 0); // pas de sens sinon...
            memset(m_host_data, value, m_size);
        }
    protected:
        const std::size_t m_size;
        uint8_t* m_host_data;
        uint8_t* m_device_data;
    };

    // @ voir les besoin en config / type
    class buffer_type : public ggml::cpp::backend::buffer_type {
        const int m_deviceId;
        const std::string m_name;
        const BUFFER_TYPE m_type;

        static const std::string name_format(int id, BUFFER_TYPE type) {
            std::ostringstream ostr;
            switch (type) {
            case BUFFER_TYPE::DEVICE:
                ostr << "IGPU_device_buffer<"<<id<<">";
                break;
            case BUFFER_TYPE::HOST:
                ostr << "IGPU_host_buffer<"<<id<<">";
                break;
            case BUFFER_TYPE::EXTRA:
                ostr << "IGPU_extra_buffer<"<<id<<">";
                break;
            }
            return ostr.str();
        }

    public:
        buffer_type(int devideId, BUFFER_TYPE type): m_deviceId(devideId), m_name(name_format(m_deviceId, type)), m_type(type) {
        }
        virtual ~buffer_type() {
            // le nom static peu avoir ete detruit...
            IGPU_TRACE("buffer_type["<< get_name() <<"]: deleted");
        }
        const std::string& get_name() override {
            return m_name;
        }
        ggml::cpp::backend::buffer* alloc_buffer(std::size_t size) override {
            IGPU_TRACE("buffer_type["<< get_name() <<"]::alloc_buffer(" << size << ")");
            // activé le GPU (si pas celui par defaut ou si plusieur.)
            ggml::hip::setDevice(m_deviceId);
            return new buffer(size, m_type);
        }
        std::size_t get_alignment() override { return TENSOR_ALIGNMENT; }
        std::size_t get_max_size() override { return SIZE_MAX; }
        std::size_t get_alloc_size(const ggml_tensor& tensor) override {
            // @ revoir si repacké/quantizé.
            return ggml_nbytes(&tensor);
        }
        bool is_host() override {
            // return true;
            return m_type != BUFFER_TYPE::EXTRA;
        }
    };

    // juste pour reference, il y a peu de chance qu'il marche pour autre chose que des float
    template<typename TA, typename TB, typename TC>
    void matmul_ref(const TA* A, const TB* B, TC* C,
            std::size_t M, std::size_t N, std::size_t K,
            std::size_t lA, std::size_t lB, std::size_t lC )
    {
        // le format de stockage natif.
        for (std::size_t i=0; i<M; i++) {
            for (std::size_t j=0; j<N; j++) {
                TC c = 0;
                for (std::size_t k=0; k<K; k++) {
                    c += ((float)A[i*lA+k])*((float)B[j*lB+k]);
                }
                C[j*lC+i] = c;
            }
        }
    }

    class backend : public ggml::cpp::backend::backend {
        const int m_deviceId;
    public:
        backend(const std::string& params, ggml::cpp::backend::device& dev, int deviceId) :
            ggml::cpp::backend::backend(dev), m_deviceId(deviceId)
        {
            IGPU_TRACE("backend[" << get_name() << "]: create <" << params << ">");
            op_mul_mat::init_caches();
        }

        virtual ~backend() {
            IGPU_TRACE("backend[" << get_name() << "]::backend deleted");
        }
        const std::string& get_name() override {
            return m_device.get_name();
        }
        ggml_guid_t get_guid() override {
            // uuidgen > afb00133-367f-4471-bdb1-530e61a00109
            static ggml_guid guid = { 0xaf, 0xb0, 0x01, 0x33, 0x36, 0x7f, 0x44, 0x71, 0xbd, 0xb1, 0x53, 0x0e, 0x61, 0xa0, 0x01, 0x09 };
            return &guid;
        }
        enum ggml_status graph_compute(ggml_cgraph & cgraph) override {
            // pour ananlyse memoire...
            // return GGML_STATUS_SUCCESS;

            // TODO activer le bon GPU...
            ggml::hip::setDevice(m_deviceId);

            for (int i = 0; i < cgraph.n_nodes; i++) {
                ggml_tensor * node = cgraph.nodes[i];
                switch (node->op) {
                case GGML_OP_MUL_MAT:
                {
                    const struct ggml_tensor * A = node->src[0];  // les poids
                    const struct ggml_tensor * B = node->src[1];
                    struct ggml_tensor * C = node;
                    const std::size_t M = C->ne[0];
                    const std::size_t N = C->ne[1];
                    const std::size_t K = A->ne[0];
                    const std::size_t la = A->nb[1]/A->nb[0];
                    const std::size_t lb = B->nb[1]/B->nb[0];
                    const std::size_t lc = C->nb[1]/C->nb[0];

                    GGML_ASSERT(K==la);
                    GGML_ASSERT(K==lb);
                    GGML_ASSERT(M==lc);

                    // calcul: op = (op->src[0]) @ op->src[1]^t;
                    // Attention cette OP n'est pas "normal" cf: https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md#coding-guidelines
                    if (A->type == GGML_TYPE_F32 && B->type == GGML_TYPE_F32 && C->type == GGML_TYPE_F32) {
                        GGML_ASSERT(A->nb[0] == sizeof(float32_t));
                        GGML_ASSERT(B->nb[0] == sizeof(float32_t));
                        GGML_ASSERT(C->nb[0] == sizeof(float32_t));
                        GGML_ASSERT(op_mul_mat::compute((const float32_t*)A->data, (const float32_t*)B->data, (float32_t*)C->data, M,N,K, la,lb,lc));
                        //matmul_ref((float32_t*)A->data, (float32_t*)B->data, (float32_t*)C->data, M,N,K, la,lb,lc);
                    }
                    if (A->type == GGML_TYPE_BF16 && B->type == GGML_TYPE_F32 && C->type == GGML_TYPE_F32) {
                        GGML_ASSERT(A->nb[0] == sizeof(bfloat16_t));
                        GGML_ASSERT(B->nb[0] == sizeof(float32_t));
                        GGML_ASSERT(C->nb[0] == sizeof(float32_t));
                        GGML_ASSERT(op_mul_mat::compute((const bfloat16_t*)A->data, (const float32_t*)B->data, (float32_t*)C->data, M,N,K, la,lb,lc));
                        //matmul_ref((bfloat16_t*)A->data, (float32_t*)B->data, (float32_t*)C->data, M,N,K, la,lb,lc);
                    }
                    if (A->type == GGML_TYPE_F16 && B->type == GGML_TYPE_F32 && C->type == GGML_TYPE_F32) {
                        GGML_ASSERT(A->nb[0] == sizeof(float16_t));
                        GGML_ASSERT(B->nb[0] == sizeof(float32_t));
                        GGML_ASSERT(C->nb[0] == sizeof(float32_t));
                        GGML_ASSERT(op_mul_mat::compute((const float16_t*)A->data, (const float32_t*)B->data, (float32_t*)C->data, M,N,K, la,lb,lc));
                        //matmul_ref((float16_t*)A->data, (float32_t*)B->data, (float32_t*)C->data, M,N,K, la,lb,lc);
                    }
                    // prevu / possible (suivant la version) pas utilisé?
                    if (A->type == GGML_TYPE_BF16 && B->type == GGML_TYPE_BF16 && C->type == GGML_TYPE_F32) {
                        GGML_ASSERT(A->nb[0] == sizeof(bfloat16_t));
                        GGML_ASSERT(B->nb[0] == sizeof(bfloat16_t));
                        GGML_ASSERT(C->nb[0] == sizeof(float32_t));
                        GGML_ASSERT(op_mul_mat::compute((const bfloat16_t*)A->data, (const bfloat16_t*)B->data, (float32_t*)C->data, M,N,K, la,lb,lc));
                        //matmul_ref((bfloat16_t*)A->data, (float32_t*)B->data, (bfloat16_t*)C->data, M,N,K, la,lb,lc);
                    }
                    if (A->type == GGML_TYPE_F16 && B->type == GGML_TYPE_F16 && C->type == GGML_TYPE_F32) {
                        GGML_ASSERT(A->nb[0] == sizeof(float16_t));
                        GGML_ASSERT(B->nb[0] == sizeof(float16_t));
                        GGML_ASSERT(C->nb[0] == sizeof(float32_t));
                        GGML_ASSERT(op_mul_mat::compute((const float16_t*)A->data, (const float16_t*)B->data, (float32_t*)C->data, M,N,K, la,lb,lc));
                        //matmul_ref((float16_t*)A->data, (float16_t*)B->data, (float32_t*)C->data, M,N,K, la,lb,lc);
                    }
                }
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
    };

    class device : public ggml::cpp::backend::device {
        const std::string m_name;
        const std::string m_desc;
        const int m_id;
        backend* m_backend = nullptr;
        buffer_type* m_extra_buffer_type;
        buffer_type* m_device_buffer_type;
        buffer_type* m_host_buffer_type;

    public:
        device(const std::string& name, int deviceId, const std::string& desc = "...") : m_name(name), m_desc(desc), m_id(deviceId) {
            // qq elements local ? buffer_type / backend si pas global!
            GGML_LOG_INFO("ggml-igpu: device[%s::%d] added: %s\n", m_name.c_str(), deviceId, m_desc.c_str());
            IGPU_TRACE("device[" << m_name <<"::" <<m_id<< "] added: <" <<m_desc<< ">");
            // Lié au device puisse que c'est lui qui alloue les buffer sur/pour le bon device!
            m_extra_buffer_type  = new buffer_type(m_id, BUFFER_TYPE::EXTRA);
            m_device_buffer_type = new buffer_type(m_id, BUFFER_TYPE::DEVICE);
            m_host_buffer_type   = new buffer_type(m_id, BUFFER_TYPE::HOST);
        }
        virtual ~device() {
            IGPU_TRACE("device[" << m_name << "] deleted");
            // delete_backend();
            // TODO: il faudrait aussi detruire les wrappers!
            delete m_extra_buffer_type;  m_extra_buffer_type=nullptr;
            delete m_device_buffer_type; m_device_buffer_type=nullptr;
            delete m_host_buffer_type;   m_host_buffer_type=nullptr;
        }
        const std::string& get_name() override {
            return m_name;
        }
        const std::string& get_description() override {
            return m_desc;
        }
        void get_memory(std::size_t & free, std::size_t & total) override {
            // ?? peut-on recuperer la memoire accessible par l'iGPU? / GTT / VRAM / RAM ?
            // doit retourner les taille memoire libre et total...
            free = 64424509440;
            total = 64424509440;
            return;
            //
            //  https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory.html#ga311c3e246a21590de14478b8bd063be2
            ggml::hip::setDevice(m_id);
            HIP_CHECK_ERROR(hipMemGetInfo(&free, &total)); // ca donne le VRAM + GTT (au moins avec le kernel 6.12+?)
            IGPU_TRACE("device[" << m_name << "] get_memory: " << free << "/" << total);
        }
        enum ggml_backend_dev_type get_type() override {
            //return GGML_BACKEND_DEVICE_TYPE_CPU;  => NON!
            //return GGML_BACKEND_DEVICE_TYPE_ACCEL;  // @ utiliser ! dans tous les cas il faut ajouter les extre_buffer
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        }
        ggml::cpp::backend::backend& init_backend(const std::string& params) override {
            // appelé quand nouveau modele?
            IGPU_TRACE("device[" << m_name << "] init_backend: " << params);
            //delete_backend();
            // si il y a qq-chose a faire coté device physique c'est plutot dans le backend?
            m_backend = new backend(params, *this, m_id);
            return *m_backend;
        }
        ggml::cpp::backend::buffer_type& get_buffer_type() override {
            // ?? comment distinguer/gerer les buffer pour les poids et les autres?
            //  => coté BACKEND_CPU => extra_buffer pour les poids!
            // a HACK for test => we need to have extra_buffer !
            if (m_backend) {
                // en attandant de faire les extras buffer.
                return *m_device_buffer_type;
            }
            return *m_extra_buffer_type;
        }

        // TODO: retourner une REF !
        bool caps_host_buffer() override { return true; }
        ggml::cpp::backend::buffer_type* get_host_buffer_type() override {
            // IGPU_TRACE(" ####################  device[" << m_name << "] get_host_buffer_type!");
            return m_host_buffer_type;
        }

        // TODO: retourner une REF !
        // buffer depuis un mmap si possible/besoin...
        //buffer* buffer_from_host_ptr(void * ptr, std::size_t size, std::size_t max_tensor_size) override { return nullptr; }
        // std::vector<buffer_type> get_extra_bufts() override ; ???

        bool supports_op(const ggml_tensor & op) override {
#ifdef DEV_ACTIVE
            {
                //   GGML_OP_UNARY => const enum ggml_unary_op op = ggml_get_unary_op(dst);  /     GGML_API const char * ggml_unary_op_name(enum ggml_unary_op op);
                // histoire de lister toutes les OPs...
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

            switch (op.op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                return true;
                //return false;
            case GGML_OP_MUL_MAT:
            {
                const struct ggml_tensor * A = op.src[0];  // les poids
                const struct ggml_tensor * B = op.src[1];  // l'entrée
                const struct ggml_tensor * C = &op;        // la sortie

                if (!ggml_is_contiguous(A)) return false;
                if (!ggml_is_contiguous(B)) return false;
                // if (!ggml_is_contiguous(op)) return false;
                // les produit simple:
                if (A->ne[2]!=1 || A->ne[3]!=1) return false;
                switch (B->type) {
                case GGML_TYPE_F32:
                    switch (A->type) {
                    case GGML_TYPE_F32:
                        return op_mul_mat::supported<float32_t,  float32_t, float32_t>(*A,*B,*C);
                    case GGML_TYPE_BF16:
                        return op_mul_mat::supported<bfloat16_t, float32_t, float32_t>(*A,*B,*C);
                    case GGML_TYPE_F16:
                        return op_mul_mat::supported<float16_t,  float32_t, float32_t>(*A,*B,*C);
                    default:
                        return false;
                    }
                    break;
                    case GGML_TYPE_BF16:
                        switch (A->type) {
                        case GGML_TYPE_BF16:
                            return op_mul_mat::supported<bfloat16_t, bfloat16_t, float32_t>(*A,*B,*C);
                        default:
                            return false;
                        }
                        case GGML_TYPE_F16:
                            switch (A->type) {
                            case GGML_TYPE_F16:
                                return op_mul_mat::supported<float16_t, float16_t, float32_t>(*A,*B,*C);
                            default:
                                return false;
                            }
                            default:
                                return false;
                }
            }
            default:
                return false;
            }
        }
        bool supports_buft(ggml_backend_buffer_type_t buft) override {
            // - Cas CPU: tout ceux qui sont en RAM (si on veux la faire il faudra faire une copy.)
            //return ggml_backend_buft_is_host(buft);
            // - cas IGPU: limité au siens pour l'instant, les autres sont a configurer/copier.
            if (buft->context) {
                // + nos propres buffers
                if (buft->context == (void*) m_extra_buffer_type)  return true;
                if (buft->context == (void*) m_device_buffer_type) return true;
                if (buft->context == (void*) m_host_buffer_type)   return true;
                // + les extra-buffer de ce backend quand disponible.
            }
            return false;
        }

        void release() override {
            // le backend a été supprimé => on peu faire du menage.
            IGPU_TRACE("device[" << m_name << "] release: " << m_backend);
            m_backend = nullptr;
            // ??? @ voir si il faut faire qq-chose sur les buffers? P'etre deleté autrement
        }
    private:
        void delete_backend() {
            // IGPU_TRACE("device[" << m_name << "]::backend delete? : " << m_backend);
            if (m_backend) {
                delete m_backend;
            }
        }
    };

    class reg: public ggml::cpp::backend::reg {
        const std::string m_name{"IGPU"};
        // les devices... Voir a les mettre device& ou device => mais faut verifier comment c'est detruit!
        std::vector<device> m_devices;
    public:
        reg() {
            GGML_LOG_INFO("ggml-igpu: backend[%s] create\n", m_name.c_str());
            IGPU_TRACE("register[" << m_name << "]: create");

            // - cas IGPU: selection/filtrage/existance.
            int deviceCount;
            HIP_CHECK_ERROR(hipGetDeviceCount(&deviceCount));

            // filter:
            // - Name: AMD Radeon 780M
            // - Compute Capability: 11.0
            // - Arch Name: gfx1103
            // - integrated: 1
            // - Total Global Memory: 31039 MiB   <= la GTT !!

            // - pour l'instant seulement les IGPU en RDNA3.
            for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
                hipDeviceProp_t deviceProp;
                HIP_CHECK_ERROR(hipGetDeviceProperties(&deviceProp, deviceId));
                //std::string arch {deviceProp.gcnArchName}; // == "gfx1103"
                if (deviceProp.integrated && deviceProp.major == 11) {
                    std::ostringstream name;
                    std::ostringstream desc;
                    name << m_name << "<" << m_devices.size() << ">";
                    desc << deviceProp.name << " (" << deviceProp.gcnArchName << ")";
                    m_devices.emplace_back(name.str(), deviceId, desc.str());
                }
            }

#if 0
            // Debug ? => https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/multi_device.html#multi-device
            // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/initialization_and_version.html#_CPPv426hipDeviceComputeCapabilityPiPi11hipDevice_t
            // int deviceCount;
            // hipGetDeviceCount(&deviceCount);
            std::cout << "Number of devices: " << deviceCount << std::endl;

            for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
            {
                // https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html
                hipDeviceProp_t deviceProp;
                hipGetDeviceProperties(&deviceProp, deviceId);
                std::cout << "Device " << deviceId << std::endl << " Properties:" << std::endl;
                std::cout << "  Name: " << deviceProp.name << std::endl;
                std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
                std::cout << "  Arch Name: " << deviceProp.gcnArchName << /*"/" << deviceProp.gcnArch <<*/ std::endl;
                std::cout << "  Architecture:" << std::endl;
                std::cout << "    Global Int32 Atomics: " << deviceProp.arch.hasGlobalInt32Atomics << std::endl;
                std::cout << "    Global Float Atomic Exch: " << deviceProp.arch.hasGlobalFloatAtomicExch << std::endl;
                std::cout << "    Shared Int32 Atomics: " << deviceProp.arch.hasSharedInt32Atomics << std::endl;
                std::cout << "    Shared Float Atomic Exch: " << deviceProp.arch.hasSharedFloatAtomicExch << std::endl;
                std::cout << "    Float Atomic Add: " << deviceProp.arch.hasFloatAtomicAdd << std::endl;
                std::cout << "    Global Int64 Atomics: " << deviceProp.arch.hasGlobalInt64Atomics << std::endl;
                std::cout << "    Shared Int64 Atomics: " << deviceProp.arch.hasSharedInt64Atomics << std::endl;
                std::cout << "    Doubles: " << deviceProp.arch.hasDoubles << std::endl;
                std::cout << "    Warp Vote: " << deviceProp.arch.hasWarpVote << std::endl;
                std::cout << "    Warp Ballot: " << deviceProp.arch.hasWarpBallot << std::endl;
                std::cout << "    Warp Shuffle: " << deviceProp.arch.hasWarpShuffle << std::endl;
                std::cout << "    Funnel Shift: " << deviceProp.arch.hasFunnelShift << std::endl;
                std::cout << "    Thread Fence System: " << deviceProp.arch.hasThreadFenceSystem << std::endl;
                std::cout << "    Sync Threads Ext: " << deviceProp.arch.hasSyncThreadsExt << std::endl;
                std::cout << "    Surface Funcs: " << deviceProp.arch.hasSurfaceFuncs << std::endl;
                std::cout << "    3D Grid: " << deviceProp.arch.has3dGrid << std::endl;
                std::cout << "    Dynamic Parallelism: " << deviceProp.arch.hasDynamicParallelism << std::endl;
                std::cout << "  integrated: " << deviceProp.integrated << std::endl;
                std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MiB" << std::endl;
                std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KiB" << std::endl;
                std::cout << "  L2 cache: " << deviceProp.l2CacheSize / 1024 << " KiB" << std::endl;
                std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
                std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
                std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
                std::cout << "  Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
                std::cout << "  Number of Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
                std::cout << "  Max Threads Dimensions: ["
                        << deviceProp.maxThreadsDim[0] << ", "
                        << deviceProp.maxThreadsDim[1] << ", "
                        << deviceProp.maxThreadsDim[2] << "]" << std::endl;
                std::cout << "  Max Grid Size: ["
                        << deviceProp.maxGridSize[0] << ", "
                        << deviceProp.maxGridSize[1] << ", "
                        << deviceProp.maxGridSize[2] << "]" << std::endl;
                std::cout << std::endl;
            }
#endif
        }
        virtual ~reg() {
            // destruction de device (quand on quitte l'appli)
            IGPU_TRACE("register[" << m_name << "] deleted");
        }
        const std::string& get_name() override {
            return m_name;
        }
        std::size_t get_device_count() override {
            if (m_devices.size()) return 1;
            return 0;
        }
        device& get_device(std::size_t index) override {
            GGML_ASSERT(index < get_device_count());
            return (m_devices[index]);
        }
        void * get_proc_address(const std::string& name) override {
            IGPU_TRACE("register[IGPU]: get_proc_address<" << name << ">");
            if (name=="ggml_backend_dev_get_extra_bufts") {

                // Pour gerer des buffer pour les poids a re-packé
                //  ggml_backend_buffer_type_t[]  ggml_backend_cpu_device_get_extra_buffers_type(ggml_backend_dev_t device)...
                //ggml_backend_dev_get_extra_bufts_t fct = ggml_backend_cpu_device_get_extra_buffers_type;
                //return (void *)fct;
            }
            return nullptr;
        }
    };

}

ggml_backend_reg_t ggml_backend_igpu_reg(void) {
    static ggml::backend::igpu::reg ctx;
    // si il y a des devices on retourne le backend.
    if (ctx.get_device_count() > 0) return ggml::cpp::backend::c_wrapper(&ctx);
    // pas d'IGPU => pas la peine de finaliser le backend.
    return nullptr;
}

GGML_BACKEND_DL_IMPL(ggml_backend_igpu_reg)
