#include "imp.h"

#include "ggml-log.h"
#include "hip-tools.h"

#include "tools.h"
// #include "types.h"

static constexpr bool NOT_IMPEMENTED = true;

/*
on va pour faciliter l'implementation se limiter au APU AMD.
- les code pour les GPU utilisent HIP
- les buffers sont accessible par l'hote
  => le backend CPU est utilisable pour toutes les OP non encore implementé sans copie.

- les OP ooptimisé qui demande un repack/retyping seront creer avec des extra-buffer
  (1 Type par OP forcement voir plus...)
- les buffer de type device sont tous de type hote
 */

namespace ggml::backend::igpu {

    static bool IS_WEIGHT = true;  // TODO plutot is repacked / type de repack ?
    static bool IS_OTHER = true;
    enum class BUFFER_TYPE {
        HOST,
        DEVICE,
        EXTRA  // @ voir la vrai liste
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
    protected:
        const BUFFER_TYPE m_type;
        const int m_id;
        const std::size_t m_size;
    public:
        buffer(BUFFER_TYPE type, std::size_t size) : m_type(type), m_id(NEXT_BUFFER_ID++), m_size(size) { }
    };

    class device_buffer : public buffer {
    public:
        device_buffer(std::size_t size) : buffer(BUFFER_TYPE::DEVICE, size) {
            GGML_ASSERT(NOT_IMPEMENTED);
        }
    };

    class host_buffer : public buffer {
    private:
        uint8_t* m_host_data = nullptr;
        uint8_t* m_device_data = nullptr;

    public:
        host_buffer(std::size_t size) : buffer(BUFFER_TYPE::HOST, size) {
            IGPU_TRACE("host_buffer[IGPU<"<<m_id<<">]::new(" << size << ", " << to_char(m_type) << ")");
            // - cas HIP/IGPU
            m_host_data = ggml::hip::allocateHost<uint8_t>(m_size);
            m_device_data = ggml::hip::getDeviceMem(m_host_data);
        }
        virtual ~host_buffer() {
            IGPU_TRACE("buffer[IGPU<"<<m_id<<">]: deleted");
            ggml::hip::deallocateHost(m_host_data);
        }
        void* get_base() override {
            return m_host_data;
        }
        ggml_status init_tensor(ggml_tensor & tensor) override {
            // const auto K = tensor.ne[0];
            // const auto M = tensor.ne[1];
            //  tensor->buffer.usage
            IGPU_TRACE("init_tensor[IGPU<"<<m_id<<":"<<tensor.buffer->usage<<">]<" << tensor.name << ">"
                       "["<<tensor.ne[0]<<", "<<tensor.ne[1]<<", "<<tensor.ne[2]<<", "<<tensor.ne[3]<<"]"
                      );

            // Permet si besoin de definir qq-chose sur extra.
            GGML_ASSERT(tensor.extra == nullptr);
            return GGML_STATUS_SUCCESS;
        }
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
            memset(m_host_data, value, m_size);
        }
    };

    // @ voir les besoin en config / type
    // TODO: gerer plusieur class (1 par type)
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
            default :
                ostr << "IGPU_???_buffer<"<<id<<">";
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
            return new host_buffer(size);
        }
        std::size_t get_alignment() override { return TENSOR_ALIGNMENT; }
        std::size_t get_max_size() override { return SIZE_MAX; }
        std::size_t get_alloc_size(const ggml_tensor& tensor) override {
            // @ revoir si repacké/quantizé.
            return ggml_nbytes(&tensor);
        }
        bool is_host() override {
            return true;
            // return m_type != BUFFER_TYPE::EXTRA;
        }
    };

    class backend;

    class device : public ggml::cpp::backend::device {
        const std::string m_name;
        const std::string m_desc;
        const int m_id;

        buffer_type* m_device_buffer_type;
        buffer_type* m_host_buffer_type;

    public:
        device(const std::string& name, int deviceId, const std::string& desc = "...") : m_name(name), m_desc(desc), m_id(deviceId) {
            // qq elements local ? buffer_type / backend si pas global!
            GGML_LOG_INFO("ggml-igpu: device[%s::%d] added: %s\n", m_name.c_str(), deviceId, m_desc.c_str());
            IGPU_TRACE("device[" << m_name <<"::" <<m_id<< "] added: <" <<m_desc<< ">");
            // Lié au device puisse que c'est lui qui alloue les buffer sur/pour le bon device!
            m_device_buffer_type = new buffer_type(m_id, BUFFER_TYPE::DEVICE);
            m_host_buffer_type   = new buffer_type(m_id, BUFFER_TYPE::HOST);
        }
        virtual ~device() {
            IGPU_TRACE("device[" << m_name << "] deleted");
            // TODO: il faudrait aussi detruire les wrappers!
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
            // TODO voir quoi retourner:
            //   sur les IGPU plutot la RAM vu que l'on alloue dessus (mais comment?)
            free = 100000000000;
            total = 100000000000;
            return;
            //
            //  https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___memory.html#ga311c3e246a21590de14478b8bd063be2
            ggml::hip::setDevice(m_id);
            HIP_CHECK_ERROR(hipMemGetInfo(&free, &total)); // ca donne le VRAM + GTT (au moins avec le kernel 6.12+?)
            IGPU_TRACE("device[" << m_name << "] get_memory: " << free << "/" << total);
        }
        enum ggml_backend_dev_type get_type() override {
            //return GGML_BACKEND_DEVICE_TYPE_CPU;    => NON!
            //return GGML_BACKEND_DEVICE_TYPE_ACCEL;  => NON: => utiliser les extre_buffer
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        }
        ggml::cpp::backend::backend& init_backend(const std::string& params) override;
        ggml::cpp::backend::buffer_type& get_buffer_type() override {
            return *m_device_buffer_type;
        }

        // TODO: retourner une REF !
        bool caps_host_buffer() override { return true; }
        ggml::cpp::backend::buffer_type* get_host_buffer_type() override {
            return m_host_buffer_type;
        }

        bool supports_op(const ggml_tensor & op) override {
            return imp::supports_op(op);
        }
        bool supports_buft(ggml_backend_buffer_type_t buft) override {
            // - Cas CPU: tout ceux qui sont en RAM (si on veux la faire il faudra faire une copy.)
            //return ggml_backend_buft_is_host(buft);
            // - cas IGPU: limité au siens pour l'instant, les autres sont a configurer/copier.
            if (buft->context) {
                // + nos propres buffers
                if (buft->context == (void*) m_device_buffer_type) return true;
                if (buft->context == (void*) m_host_buffer_type)   return true;
                // + les extra-buffer de ce backend quand disponible.
            }
            return false;
        }
        int getID() const {return m_id;}
    };

    class backend : public ggml::cpp::backend::backend {
        const int m_deviceId;
    public:
        backend(const std::string& params, device& dev) :
            ggml::cpp::backend::backend(dev), m_deviceId(dev.getID())
        {
            IGPU_TRACE("backend[" << get_name() << "]: create <" << params << ">");
            // TODO: les inits
        }

        virtual ~backend() {
            IGPU_TRACE("backend[" << get_name() << "]::backend deleted");
            // Note[JPP]: c'est celui qui le recupere qui demande le delete.
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
            // pour analyse memoire...
            // return GGML_STATUS_SUCCESS;
            // - activer le bon GPU...
            ggml::hip::setDevice(m_deviceId);
            auto res = imp::graph_compute(cgraph);
            HIP_CHECK_ERROR(hipDeviceSynchronize());
            return res;
        }
    };

    ggml::cpp::backend::backend& device::init_backend(const std::string& params) {
        // recuperer pour gerer un device, et detruit par l'appelant.
        //  c'est plus un new que un init.  => @ voir si on n'en fait pas un pointer!
        auto back = new backend(params, *this);
        return *back;
    }

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
                    // TODO: voir quel autre elements recuperer (APU/eGPU, arch supportée ... RDNA1/2/3/4...)
                    m_devices.emplace_back(name.str(), deviceId, desc.str());
                    IGPU_TRACE("register device:" << name << "(" << deviceProp.major <<":"<<deviceProp.minor << ")");
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
