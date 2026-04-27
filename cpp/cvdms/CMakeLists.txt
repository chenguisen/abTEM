cmake_minimum_required(VERSION 3.21)

# Use GCC 12 for both host and CUDA (LTO compatibility with CUDA 12.0)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-12 CACHE FILEPATH "CUDA host compiler")
set(CMAKE_CXX_COMPILER /usr/bin/g++-12 CACHE FILEPATH "C++ compiler")

project(_cvdms_backend LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_FLAGS "-O3 --default-stream per-thread")

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ──────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────
set(CVDMS_SOURCES
    src/Laplacian.cu
    src/KOperator.cu
    src/KSeries.cu
    src/TaylorSeries.cu
    src/Backscattering.cu
    src/Convergence.cu
    bindings/module.cpp
)

pybind11_add_module(_cvdms_backend MODULE ${CVDMS_SOURCES})

# Install as top-level module (import _cvdms_backend)
install(TARGETS _cvdms_backend LIBRARY DESTINATION .)

target_include_directories(_cvdms_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(_cvdms_backend
    PRIVATE
        CUDA::cudart
        CUDA::cufft
)
set_target_properties(_cvdms_backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
