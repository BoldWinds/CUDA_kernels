add_executable(transpose
    transpose.cu
)

target_link_libraries(transpose PRIVATE 
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    common_headers
)

set_target_properties(transpose PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)