add_executable(reduce_sum
    reduce_sum.cu
)

target_link_libraries(reduce_sum PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    common_headers
)

set_target_properties(reduce_sum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(reduce_max
    reduce_max.cu
)

target_link_libraries(reduce_max PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    common_headers
)

set_target_properties(reduce_max PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)