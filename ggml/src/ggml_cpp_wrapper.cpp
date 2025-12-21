#include "ggml_cpp_wrapper.h"

#include "ggml-backend-impl.h"

#include <map>
#include <memory>

namespace ggml::cpp::backend {

// TODO: voir si on ne cree pas une fontion static plutot que friend.
ggml_backend_buffer_type_t* backend_dev_get_extra_bufts(ggml_backend_dev_t device) {
    auto& ctx = *((ggml::cpp::backend::device*) (device->context));
    if (ctx.m_ggml_extra_buffers_type.size() == 0) { // need init of extra buffer wrappers
        for (auto* buft : ctx.m_extra_buffers_type) {
            auto c_buft = c_wrapper(device, buft);
            ctx.m_ggml_extra_buffers_type.push_back(c_buft);
        }
        ctx.m_ggml_extra_buffers_type.push_back(nullptr);
    }
    return ctx.m_ggml_extra_buffers_type.data();
}

    namespace { // unnamed namespace

    //=========================================================
    // les wrappper pour ggml_backend_buffer
    void buffer_free_buffer(ggml_backend_buffer_t buf) {
        auto* ctx = (ggml::cpp::backend::buffer*) (buf->context);
        delete ctx;
        // delete buf; NO => deleted by the core.
    }
    void * buffer_get_base(ggml_backend_buffer_t buf) {
        auto& ctx = *((ggml::cpp::backend::buffer*) (buf->context));
        return ctx.get_base();
    }
    ggml_status buffer_init_tensor(ggml_backend_buffer_t buf, ggml_tensor * tensor) {
        auto& ctx = *((ggml::cpp::backend::buffer*) (buf->context));
        return ctx.init_tensor(*tensor);
    }
    void buffer_memset_tensor(ggml_backend_buffer_t buf,       ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size) {
        auto& ctx = *((ggml::cpp::backend::buffer*) (buf->context));
        ctx.memset_tensor(*tensor, value, offset, size);
    }
    void buffer_set_tensor(ggml_backend_buffer_t buf,       ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
        auto& ctx = *((ggml::cpp::backend::buffer*) (buf->context));
        ctx.set_tensor(*tensor, data, offset, size);
    }
    void buffer_get_tensor(ggml_backend_buffer_t buf, const ggml_tensor * tensor,       void * data, size_t offset, size_t size) {
        auto& ctx = *((ggml::cpp::backend::buffer*) (buf->context));
        ctx.get_tensor(*tensor, data, offset, size);
    }
    bool buffer_cpy_tensor(ggml_backend_buffer_t buf, const ggml_tensor * src, ggml_tensor * dst) {
        auto& ctx = *((ggml::cpp::backend::buffer*) (buf->context));
        return ctx.cpy_tensor(*src, *dst);
    }
    void buffer_clear(ggml_backend_buffer_t buf, uint8_t value) {
        auto& ctx = *((ggml::cpp::backend::buffer*) (buf->context));
        ctx.clear(value);
    }
    void buffer_reset(ggml_backend_buffer_t buf) {
        auto& ctx = *((ggml::cpp::backend::buffer*) (buf->context));
        ctx.reset();
    }

    //=========================================================
    // wrapppers for ggml_backend_buffer_type
    const char *buffer_type_get_name (ggml_backend_buffer_type_t buft) {
        auto& ctx = *((ggml::cpp::backend::buffer_type*) (buft->context));
        return ctx.get_name().c_str();
    }
    std::size_t buffer_type_get_alignment (ggml_backend_buffer_type_t buft) {
        auto& ctx = *((ggml::cpp::backend::buffer_type*) (buft->context));
        return ctx.get_alignment();
    }
    std::size_t buffer_type_get_max_size  (ggml_backend_buffer_type_t buft) {
        auto& ctx = *((ggml::cpp::backend::buffer_type*) (buft->context));
        return ctx.get_max_size();
    }
    std::size_t buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
        auto& ctx = *((ggml::cpp::backend::buffer_type*) (buft->context));
        return ctx.get_alloc_size(*tensor);
    }
    bool   buffer_type_is_host      (ggml_backend_buffer_type_t buft) {
        auto& ctx = *((ggml::cpp::backend::buffer_type*) (buft->context));
        return ctx.is_host();
    }
    ggml_backend_buffer_t buffer_type_alloc_buffer (ggml_backend_buffer_type_t buft, std::size_t size) {
        auto& ctx = *((ggml::cpp::backend::buffer_type*) (buft->context));
        return c_wrapper(buft, ctx.alloc_buffer(size), size);
    }

    //=========================================================
    // wrapppers for ggml_backend
    const char * backend_get_name(ggml_backend_t bkd) {
        auto& ctx = *((ggml::cpp::backend::backend*) (bkd->context));
        return ctx.get_name().c_str();
    }
    void backend_free(ggml_backend_t backend) {
        // TODO: figure the best to delete the backend.
        //  - managed by the device
        //  - deleted here
        auto* ctx = (ggml::cpp::backend::backend*) (backend->context);
        delete ctx;
        delete backend;
    }
    void backend_set_tensor_async(ggml_backend_t bkd,       ggml_tensor * tensor, const void * data, std::size_t offset, std::size_t size) {
        auto& ctx = *((ggml::cpp::backend::backend*) (bkd->context));
        ctx.set_tensor_async(*tensor, data, offset, size);
    }
    void backend_get_tensor_async(ggml_backend_t bkd, const ggml_tensor * tensor,       void * data, std::size_t offset, std::size_t size) {
        auto& ctx = *((ggml::cpp::backend::backend*) (bkd->context));
        ctx.get_tensor_async(*tensor, data, offset, size);
    }
    bool backend_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
        auto& ctx = *((ggml::cpp::backend::backend*) (backend_dst->context));
        return ctx.cpy_tensor_async(backend_src, *src, *dst);
    }
    void backend_synchronize(ggml_backend_t bkd) {
        auto& ctx = *((ggml::cpp::backend::backend*) (bkd->context));
        ctx.synchronize();
    }
    enum ggml_status backend_graph_compute(ggml_backend_t bkd, ggml_cgraph * cgraph) {
        auto& ctx = *((ggml::cpp::backend::backend*) (bkd->context));
        return ctx.graph_compute(*cgraph);
    }
    void backend_event_record(ggml_backend_t bkd, ggml_backend_event_t evt) {
        auto& ctx = *((ggml::cpp::backend::backend*) (bkd->context));
        ctx.event_record(*((event*) evt));
    }
    void backend_event_wait  (ggml_backend_t bkd, ggml_backend_event_t evt) {
        auto& ctx = *((ggml::cpp::backend::backend*) (bkd->context));
        ctx.event_wait(*((event*) evt));
    }
    void backend_set_n_threads(ggml_backend_t bkd, int n_threads) {
        auto& ctx = *((ggml::cpp::backend::backend*) (bkd->context));
        ctx.set_n_threads(n_threads);
    }

    //=========================================================
    // wrapppers for ggml_backend_device
    const char * device_get_name(ggml_backend_dev_t dev) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        return ctx.get_name().c_str();
    }
    const char * device_get_description(ggml_backend_dev_t dev) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        return ctx.get_description().c_str();
    }
    void device_get_memory(ggml_backend_dev_t dev, std::size_t * free, std::size_t * total) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        ctx.get_memory(*free, *total);
    }
    enum ggml_backend_dev_type device_get_type(ggml_backend_dev_t dev) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        return ctx.get_type();
    }
    void device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        props->name = ctx.get_name().c_str();
        props->description = ctx.get_description().c_str();
        ctx.get_memory(props->memory_free, props->memory_total);
        props->type = ctx.get_type();
        props->caps.async = ctx.caps_async();
        props->caps.host_buffer = ctx.caps_host_buffer();
        props->caps.buffer_from_host_ptr = ctx.caps_buffer_from_host_ptr();
        props->caps.events = ctx.caps_events();
    }
    ggml_backend_t device_init_backend(ggml_backend_dev_t dev, const char * params) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        return c_wrapper(dev, &ctx.init_backend(params?params:""));
    }
    ggml_backend_buffer_type_t device_get_buffer_type(ggml_backend_dev_t dev) {
        // Note: nothing to delete it.
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        return c_wrapper(dev, &ctx.get_buffer_type());
    }
    ggml_backend_buffer_type_t device_get_host_buffer_type(ggml_backend_dev_t dev) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        auto* bft = ctx.get_host_buffer_type();
        if (bft) {
            return c_wrapper(dev, bft);
        }
        return nullptr;
    }
    ggml_backend_buffer_t device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, std::size_t size, std::size_t max_tensor_size) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        auto* bft = ctx.get_from_host_ptr_buffer_type();
        if (!bft) { return nullptr; }
        auto* buf = bft->register_buffer(ptr, size, max_tensor_size);
        if (!buf) { return nullptr; }
        // comment / ou memoriser ce wrapper, il n'y a pas de "delete"
        auto * ggml_buf_type = c_wrapper(dev, bft);
        return c_wrapper(ggml_buf_type, buf, size);
    }
    bool device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        return ctx.supports_op(*op);
    }
    bool device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        return ctx.supports_buft(buft /*->context*/);
    }
    bool device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        return ctx.offload_op(*op);
    }
    ggml_backend_event_t device_event_new (ggml_backend_dev_t dev) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        auto* evt = ctx.event_new();
        if (!evt) { return nullptr; }
        return new ggml_backend_event {
            dev,
            evt,
        };
    }

    void device_event_free(ggml_backend_dev_t /*dev*/, ggml_backend_event_t evt_c) {
        auto* evt_cpp = (event*)(evt_c->context);
        delete evt_cpp;
        delete evt_c;
    }

    void device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t evt_c) {
        auto& ctx = *((ggml::cpp::backend::device*) (dev->context));
        auto* evt_cpp = (event*)(evt_c->context);
        ctx.event_synchronize(*evt_cpp);
    }

    //=========================================================
    // wrapppers for ggml_backend_reg
    const char *       reg_get_name(ggml_backend_reg_t reg) {
        auto& ctx = *((ggml::cpp::backend::reg*) (reg->context));
        return ctx.get_name().c_str();
    }
    std::size_t        reg_get_device_count(ggml_backend_reg_t reg) {
        auto& ctx = *((ggml::cpp::backend::reg*) (reg->context));
        return ctx.get_device_count();
    }
    ggml_backend_dev_t reg_get_device(ggml_backend_reg_t reg, std::size_t index) {
        auto& ctx = *((ggml::cpp::backend::reg*) (reg->context));
        return c_wrapper(reg, &ctx.get_device(index));
    }
    void * reg_get_proc_address(ggml_backend_reg_t /*reg*/, const char * cname) {
        const auto name = std::string(cname);
        if (name == "ggml_backend_set_n_threads") {
            return (void *)backend_set_n_threads;
        }
        if (name == "ggml_backend_dev_get_extra_bufts") {
            return (void*) backend_dev_get_extra_bufts;
        }
        return nullptr;
    }

    }

    // les destructeurs...
    buffer::~buffer() {}
    buffer_type::~buffer_type() {}
    event::~event() {}
    backend::backend(device& dev): m_device(dev) {}
    backend::~backend() { }
    device::~device() {}
    reg::~reg() {}

    // non virtual fct:
    void device::register_extra_buffer_type(buffer_type* buft) {
        GGML_ASSERT(m_ggml_extra_buffers_type.size() == 0); // pas encore initialisé!
        m_extra_buffers_type.push_back(buft);
    }

    //=========================================================
    // the wrappers
    ggml_backend_buffer_t c_wrapper(ggml_backend_buffer_type_t buft, buffer* ctx, std::size_t size) {
        if (!ctx) { return nullptr; }
        return new ggml_backend_buffer {
            /* .interface = */ {
                /* .free_buffer     = */ buffer_free_buffer,
                /* .get_base        = */ buffer_get_base,
                /* .init_tensor     = */ buffer_init_tensor,
                /* .memset_tensor   = */ buffer_memset_tensor,
                /* .set_tensor      = */ buffer_set_tensor,
                /* .get_tensor      = */ buffer_get_tensor,
                /* .cpy_tensor      = */ buffer_cpy_tensor,
                /* .clear           = */ buffer_clear,
                /* .reset           = */ buffer_reset,
            },
            /* .buft      = */ buft,
            /* .context   = */ ctx,
            /* .size      = */ size,
            /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
        };
    }

    struct buffer_type_deleter {
        void operator()(ggml_backend_buffer_type* c_buffer_type) {
            // do something with buffer_type::c_buffer_type.context ?
            delete (c_buffer_type);
        }
    };
    typedef std::unique_ptr<ggml_backend_buffer_type,  buffer_type_deleter>  c_buffer_type_ptr;

    ggml_backend_buffer_type_t c_wrapper(ggml_backend_dev_t device, buffer_type* ctx) {
        // the ctx have to be "static".
        static std::map<buffer_type*, c_buffer_type_ptr> map;
        // static std::map<buffer_type*, ggml_backend_buffer_type_t> map;
        if (!ctx) { return nullptr; }

        auto it = map.find(ctx);
        // add new wrapper if not find.
        if (it == map.end()) {
            auto wrapper = new ggml_backend_buffer_type {
                /* .iface    = */ {
                    /* .get_name         = */ buffer_type_get_name,
                    /* .alloc_buffer     = */ buffer_type_alloc_buffer,
                    /* .get_alignment    = */ buffer_type_get_alignment,
                    /* .get_max_size     = */ buffer_type_get_max_size,
                    /* .get_alloc_size   = */ buffer_type_get_alloc_size,
                    /* .is_host          = */ buffer_type_is_host,
                },
                /* .device  = */ device, // ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
                /* .context = */ ctx,
            };
            map[ctx] = c_buffer_type_ptr(wrapper);
            //map[ctx] = wrapper;
            return wrapper;
        }
        return it->second.get();
        //return it->second;
    }

    ggml_backend_t c_wrapper(ggml_backend_dev_t device, backend* ctx) {
        if (!ctx) { return nullptr; }
        auto& dev = *((ggml::cpp::backend::device*) (device->context));
        return new ggml_backend {
            /* .guid    = */ ctx->get_guid(),
            /* .iface   = */ {
                /* .get_name           = */ backend_get_name,
                /* .free               = */ backend_free,
                /* .set_tensor_async   = */ dev.caps_async() ? backend_set_tensor_async : nullptr,
                /* .get_tensor_async   = */ dev.caps_async() ? backend_get_tensor_async : nullptr,
                /* .cpy_tensor_async   = */ dev.caps_async() ? backend_cpy_tensor_async : nullptr,
                /* .synchronize        = */ dev.caps_async() ? backend_synchronize : nullptr,
                /* .graph_plan_create  = */ nullptr,
                /* .graph_plan_free    = */ nullptr,
                /* .graph_plan_update  = */ nullptr,
                /* .graph_plan_compute = */ nullptr,
                /* .graph_compute      = */ backend_graph_compute,
                /* .event_record       = */ dev.caps_events() ? backend_event_record : nullptr,
                /* .event_wait         = */ dev.caps_events() ? backend_event_wait : nullptr,
                /* .graph_optimize     = */ nullptr, // TODO: pour fusion...etc 
            },
            /* .device  = */ device,
            /* .context = */ ctx
        };
    }

    struct device_deleter {
        void operator()(ggml_backend_device* c_device) {
            // do something with device::c_device.context ?
            delete (c_device);
        }
    };
    typedef std::unique_ptr<ggml_backend_device,  device_deleter>  c_device_ptr;

    ggml_backend_dev_t c_wrapper(ggml_backend_reg_t reg, device* ctx) {
        // the ctx have to be "static" / "per backend_register"
        static std::map<device*, c_device_ptr> map;
        if (!ctx) { return nullptr; }

        auto it = map.find(ctx);
        if (it == map.end()) {
            auto wrapper = new ggml_backend_device {
                /* .iface       = */{
                    /* .get_name             = */ device_get_name,
                    /* .get_description      = */ device_get_description,
                    /* .get_memory           = */ device_get_memory,
                    /* .get_type             = */ device_get_type,
                    /* .get_props            = */ device_get_props,
                    /* .init_backend         = */ device_init_backend,
                    /* .get_buffer_type      = */ device_get_buffer_type,
                    /* .get_host_buffer_type = */ ctx->caps_host_buffer() ? device_get_host_buffer_type : nullptr,
                    /* .buffer_from_host_ptr = */ ctx->caps_buffer_from_host_ptr() ? device_buffer_from_host_ptr : nullptr,
                    /* .supports_op          = */ device_supports_op,
                    /* .supports_buft        = */ device_supports_buft,
                    /* .offload_op           = */ device_offload_op,
                    /* .event_new            = */ ctx->caps_events() ? device_event_new : nullptr,
                    /* .event_free           = */ ctx->caps_events() ? device_event_free : nullptr,
                    /* .event_synchronize    = */ ctx->caps_events() ? device_event_synchronize : nullptr,
                },
                /* .reg         = */ reg,
                /* .context     = */ ctx,
            };
            map[ctx] = c_device_ptr(wrapper);
            return wrapper;
        }
        return it->second.get();
    }

    struct register_deleter {
        void operator()(ggml_backend_reg_t c_register) {
            delete (c_register);
        }
    };
    typedef std::unique_ptr<ggml_backend_reg,  register_deleter>  c_register_ptr;

    ggml_backend_reg_t c_wrapper(reg* ctx) {
        // the ctx have to be static.
        static std::map<reg*, c_register_ptr> map;
        if (!ctx) { return nullptr; }

        auto it = map.find(ctx);
        if (it == map.end()) {
            auto wrapper = new ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ {
                    /* .get_name         = */ reg_get_name,
                    /* .get_device_count = */ reg_get_device_count,
                    /* .get_device       = */ reg_get_device,
                    /* .get_proc_address = */ reg_get_proc_address,
                },
                /* .context     = */ ctx,
            };
            map[ctx] = c_register_ptr(wrapper);
            return wrapper;
        }
        return it->second.get();
    }

}
