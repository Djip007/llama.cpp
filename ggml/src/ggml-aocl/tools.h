#pragma once

//#define TRACE_ACTIVE
//#define DEV_ACTIVE

#ifdef TRACE_ACTIVE
#include <iostream>
#define AOCL_TRACE(...) std::cout << "#> ggml-aocl: " << __VA_ARGS__ << std::endl
#else
#define AOCL_TRACE(...)
#endif

#ifdef DEV_ACTIVE
#include <iostream>
//#include  <sstream>
#define AOCL_DEV(...) std::cout << "#> ggml-aocl: " << __VA_ARGS__ << std::endl
#else
#define AOCL_DEV(...)
#endif

#define AOCL_INFO(...) std::cout << "#> ggml-aocl: " << __VA_ARGS__ << std::endl
