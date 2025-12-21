cmake_minimum_required(VERSION 3.26)

# TODO?
if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for AOCL")
endif()

find_package(OpenMP REQUIRED)

# Find AOCL-DLP package
include(FetchContent)

# it do not work: see https://github.com/amd/aocl-dlp
#    use -DCMAKE_PREFIX_PATH="<aocl-dlp install path>"
# https://github.com/amd/aocl-dlp/releases/tag/AOCL-Weekly-051225
# GIT_TAG 944f859825b9be4c53d67c358dbdbfbd3bcb98ba # tag/AOCL-Weekly-051225
# GIT_TAG dev                                      # branch
#  https://github.com/Djip007/aocl-dlp : have a simple patch but need more.
#  https://github.com/amd/aocl-dlp.git
FetchContent_Declare(AoclDlp
                     GIT_REPOSITORY https://github.com/Djip007/aocl-dlp.git
                     GIT_TAG        dev
                     CMAKE_CACHE_ARGS -DDLP_THREADING_MODEL:STRING=openmp
                     FIND_PACKAGE_ARGS
                     )
FetchContent_MakeAvailable(AoclDlp)

set(TARGET_NAME ggml-aocl)

ggml_add_backend_library(${TARGET_NAME}
    ../../include/ggml-aocl.h
    ggml-aocl.cpp
)

add_compile_definitions(GGML_USE_AOCL)

target_compile_features(${TARGET_NAME} PRIVATE c_std_11 cxx_std_20)

target_compile_options(${TARGET_NAME} PRIVATE ${OpenMP_CXX_FLAGS} -march=native -O3)

target_link_libraries(${TARGET_NAME} PRIVATE 
    ggml-base
    OpenMP::OpenMP_CXX 
    #AoclDlp::aocl-dlp
    AoclDlp::aocl-dlp_static
)

