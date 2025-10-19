if (WIN32)
    message(FATAL_ERROR "Not teste on Windows OS")
endif()

# pas encore bon dans 
# => /usr/share/cmake/Modules/CMakeDetermineHIPCompiler.cmake
# => /usr/lib64/cmake/hip-lang/hip-lang-config.cmake vs /usr/local/lib64/cmake/hip-lang/hip-lang-config.cmake
#   hipcc                         -v -print-targets | grep "Found HIP installation"  => mauvais?
#   /usr/lib64/llvm18/bin/clang++ -v -print-targets | grep "Found HIP installation"  => mauvais !!!
#   clang-18                      -v -print-targets | grep "Found HIP installation"  => OK
#   hipconfig --rocmpath   => OK  => on va le forcer

# forcer la dernier methode...
if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT)
  execute_process(
    COMMAND hipconfig --rocmpath
    OUTPUT_VARIABLE _CMAKE_HIPCONFIG_ROCMPATH
    RESULT_VARIABLE _CMAKE_HIPCONFIG_RESULT
    )
  if(_CMAKE_HIPCONFIG_RESULT EQUAL 0 AND EXISTS "${_CMAKE_HIPCONFIG_ROCMPATH}")
    set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_CMAKE_HIPCONFIG_ROCMPATH}")
  endif()
endif()

# Forward AMDGPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
if (AMDGPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES ${AMDGPU_TARGETS})
endif()
cmake_minimum_required(VERSION 3.21)

enable_language(HIP)

find_package(hip     REQUIRED)
find_package(OpenMP  REQUIRED)

# pas testé d'autre version... => 6.4.4 ou meme 7.0.2
if (${hip_VERSION} VERSION_LESS 6.2)
    message(FATAL_ERROR "At least ROCM/HIP V6.2 is required")
endif()

message(STATUS "HIP found")

set(TARGET_NAME ggml-igpu)

file(GLOB   GGML_SOURCES_IGPU "*.cpp")
#file(GLOB   SRCS "*.hip")
#list(APPEND GGML_SOURCES_IGPU ${SRCS})

ggml_add_backend_library(${TARGET_NAME}
                         ../../include/ggml-igpu.h
                         hip-tools.h
                         hip-type.h
                         imp.h
                         ops.h
                         tools.h
                         types.h
                         ${GGML_SOURCES_IGPU}
                        )

add_compile_definitions(GGML_USE_IGPU)

# @ voir a limiter au fichier qui l'utilise vraiement...
set_source_files_properties(${GGML_SOURCES_IGPU} PROPERTIES LANGUAGE HIP)

target_compile_features(${TARGET_NAME} PRIVATE c_std_11 cxx_std_20)

target_compile_options(${TARGET_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(${TARGET_NAME} PRIVATE hip::device ${OpenMP_CXX_FLAGS})

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()


target_link_libraries(${TARGET_NAME} PRIVATE ggml-base hip::host OpenMP::OpenMP_CXX ${OpenMP_CXX_FLAGS})

message(STATUS "OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS} ")
