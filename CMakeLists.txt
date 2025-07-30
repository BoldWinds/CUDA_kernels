cmake_minimum_required(VERSION 3.18)

project(CUDA_kernels
        LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


include_directories(include)

file(GLOB_RECURSE SOURCE_FILES
    "src/*.cpp"
    "src/*.cu"
)


set(EXECUTABLE_NAME cuda_kernels)

add_executable(${EXECUTABLE_NAME} ${SOURCE_FILES})


target_include_directories(${EXECUTABLE_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${CUDA_LIBRARIES})

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# 提示用户如何构建
message(STATUS "Configuration complete. To build, run:")
message(STATUS "  cd build")
message(STATUS "  cmake ..")
message(STATUS "  make")