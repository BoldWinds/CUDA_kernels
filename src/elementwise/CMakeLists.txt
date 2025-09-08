add_executable(elementwise
    elementwise.cu
)

target_link_libraries(elementwise PRIVATE 
    CUDA::cudart
    CUDA::curand
    common_headers
)

set_target_properties(elementwise PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)