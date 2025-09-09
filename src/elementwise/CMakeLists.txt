add_executable(elementwise_add
    elementwise_add.cu
)

target_link_libraries(elementwise_add PRIVATE 
    CUDA::cudart
    CUDA::curand
    common_headers
)

set_target_properties(elementwise_add PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(sigmoid
    sigmoid.cu
)

target_link_libraries(sigmoid PRIVATE 
    CUDA::cudart
    CUDA::curand
    common_headers
)

#target_compile_options(sigmoid PRIVATE -use_fast_math)

set_target_properties(sigmoid PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)