#include "timer.h"
#include <cublas_v2.h>

__global__ void transpose_naive(float* matrix_input, float* matrix_output, unsigned rows, unsigned cols){
    unsigned row_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned col_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if(col_idx < cols && row_idx < rows) {
        matrix_output[col_idx * rows + row_idx] = matrix_input[row_idx * cols + col_idx];
    }
}

__global__ void transpose_x4(float* matrix_input, float* matrix_output, unsigned rows, unsigned cols){
    unsigned row_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned col_idx = 4 * (blockDim.y * blockIdx.y + threadIdx.y);

    if(col_idx < cols && row_idx < rows) {
        if(col_idx < cols - 3) {
            float4 reg_input = reinterpret_cast<float4*>(&matrix_input[row_idx * cols + col_idx])[0];
            matrix_output[col_idx * rows + row_idx] = reg_input.x;
            matrix_output[(col_idx + 1) * rows + row_idx] = reg_input.y;
            matrix_output[(col_idx + 2) * rows + row_idx] = reg_input.z;
            matrix_output[(col_idx + 3) * rows + row_idx] = reg_input.w;
        } else {
            for(unsigned i = 0; i < cols - col_idx; ++i) {
                matrix_output[(col_idx + i) * rows + row_idx] = matrix_input[row_idx * cols + col_idx + i];
            }
        }
        
    } 
}


int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "input max run times!" << std::endl;
        return -1;
    }
    int max_run = std::atoi(argv[1]);
    float *a, *b;
    const unsigned row = 1<<15, col = 1<<15;
    CUDA_CHECK(cudaMalloc(&a, row * col * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, row * col * sizeof(float)));
    Timer* timer = new Timer();
    dim3 threadsPerBlock(16, 16);
    double duration = 0.0;
    
    // ---------------------------------
    // CUBLAS
    // ---------------------------------
    {
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            cublasHandle_t handle;
            cublasCreate(&handle);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        col, row, &alpha, a, col,
                        &beta, NULL, row, b, row
                    );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            cublasDestroy(handle);
            timer->stop();
            duration += timer->elapsed();
        }
        std::cout << "cuBLAS: " << duration / max_run << " ms" << std::endl;
        
    }

    // ---------------------------------
    // naive
    // ---------------------------------
    {
        dim3 blocksPerGrid((row + 15)/16, (col+15)/16);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_naive<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
        }
        std::cout << "naive: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // 4x
    // ---------------------------------
    {
        dim3 blocksPerGrid((row + 15)/16, (col/4 + 15)/16);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_x4<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
        }
        std::cout << "x4: " << duration/max_run << std::endl;
        duration = 0.0;
    }


    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
}