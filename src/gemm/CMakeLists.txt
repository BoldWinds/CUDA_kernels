add_executable(sgemm
    sgemm.cu
)

target_link_libraries(sgemm PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    common_headers
)

set_target_properties(sgemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)