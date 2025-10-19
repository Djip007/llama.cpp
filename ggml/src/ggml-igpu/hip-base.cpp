#include "hip-tools.h"

namespace ggml::hip {
    void setDevice(int id) {
        HIP_CHECK_ERROR(hipSetDevice(id));
    }
}
