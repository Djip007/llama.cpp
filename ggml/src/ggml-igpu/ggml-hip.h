#pragma once
#include <hip/hip_runtime.h>

#include "ggml.h"

#define HIP_CHECK_ERROR(trt) \
    do {                                                                           \
        hipError_t _tmpVal;                                                        \
        if((_tmpVal = trt) != hipSuccess) {                                        \
           GGML_ABORT("HIP_ERROR(%s => %s)", hipGetErrorString(_tmpVal), #trt);    \
        }                                                                          \
    } while(0)

namespace ggml::hip {

    template<typename T>
    T* allocateHost(const std::size_t size) {
        void * ptr;
        HIP_CHECK_ERROR(hipHostMalloc(&ptr, size*sizeof(T), hipHostMallocNonCoherent));
        return reinterpret_cast<T*>(ptr);
    }

    template<typename T>
    T* allocateDevice(const std::size_t size) {
      void * ptr;
      HIP_CHECK_ERROR(hipMalloc(&ptr, size*sizeof(T)));
      return reinterpret_cast<T*>(ptr);
    }

    template<typename T>
    void deallocateHost(T * ptr) {
        HIP_CHECK_ERROR(hipHostFree((void*)ptr));
    }

    template<typename T>
    void deallocateDevice(T * ptr) {
      HIP_CHECK_ERROR(hipFree((void*)ptr));
    }

    template<typename T>
    T* getDeviceMem(T* host_adr) {
        void * ptr=nullptr;
        HIP_CHECK_ERROR(hipHostGetDevicePointer(&ptr, host_adr, 0));
        return reinterpret_cast<T*>(ptr);
    }

    void setDevice(int id);
}
