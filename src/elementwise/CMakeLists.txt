add_executable(elementwise
    elementwise.cu
)

target_link_libraries(elementwise PRIVATE 
    CUDA::cudart
    CUDA::curand
    common_headers
)