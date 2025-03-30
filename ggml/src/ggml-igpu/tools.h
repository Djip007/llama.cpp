#pragma once
//#define TRACE_ACTIVE
//#define DEV_ACTIVE

#ifdef TRACE_ACTIVE
#define IGPU_TRACE(...) std::cout << "#> ggml-igpu: " << __VA_ARGS__ << std::endl
#else
#define IGPU_TRACE(...)
#endif

#ifdef DEV_ACTIVE
#include  <sstream>
#define IGPU_DEV(...) std::cout << "#> ggml-igpu: " << __VA_ARGS__ << std::endl
#else
#define IGPU_DEV(...)
#endif

