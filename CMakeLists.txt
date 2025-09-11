cmake_minimum_required(VERSION 3.24)

project(CUDA_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

# public include directory
add_library(common_headers INTERFACE)
target_include_directories(common_headers INTERFACE 
    ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory(src/elementwise)
add_subdirectory(src/transpose)