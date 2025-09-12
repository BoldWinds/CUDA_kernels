#include "timer.h"
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

__global__ void verify_transpose_kernel(const float* input, const float* output, unsigned rows, unsigned cols, int* error_count) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int index_A = row * cols + col;
        int index_B = col * rows + row;

        float val_A = input[index_A];
        float val_B = output[index_B];

        if (val_A != val_B) {
            atomicAdd(error_count, 1);
        }
    }
}

void verify_transpose(const float* input, const float* output, unsigned rows, unsigned cols, int* error_count) {
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    
    CUDA_CHECK(cudaMemset(error_count, 0, sizeof(int)));
    verify_transpose_kernel<<<gridDim, blockDim>>>(input, output, rows, cols, error_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_error_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_error_count, error_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_error_count != 0) {
        std::cout << "ERROR: Verification failed! Found " << h_error_count << " mismatch(es)." << std::endl;
        std::cout << "       The provided matrix is NOT a correct transpose." << std::endl;
    }
}

// 写入是合并的
__global__ void transpose_naive(float* matrix_input, float* matrix_output, unsigned height, unsigned width){
    unsigned row_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned col_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if(col_idx < width && row_idx < height) {
        matrix_output[col_idx * height + row_idx] = matrix_input[row_idx * width + col_idx];
    }
}

__global__ void transpose_naive_64(float* matrix_input, float* matrix_output, unsigned height, unsigned width){
    unsigned row_idx = 64 * blockIdx.x + threadIdx.x;
    unsigned col_idx = 64 * blockIdx.y + threadIdx.y;

    #pragma unroll
    for(int i = 0; i < 64; i+=8){
        if(col_idx + i< width && row_idx < height){
            matrix_output[(col_idx + i) * height + row_idx] = matrix_input[row_idx * width + col_idx + i];
            if(row_idx + 32 < height) matrix_output[(col_idx + i) * height + row_idx + 32] = matrix_input[(row_idx + 32) * width + col_idx + i];
        }
    }
}

// 读取是合并的
__global__ void transpose_naive_v2(float* matrix_input, float* matrix_output, unsigned height, unsigned width){
    unsigned row_idx = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned col_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(col_idx < width && row_idx < height) {
        matrix_output[col_idx * height + row_idx] = matrix_input[row_idx * width + col_idx];
    }
}

// 写入合并
__global__ void transpose_naive_1D(float* matrix_input, float* matrix_output, const unsigned height, const unsigned width){
    const unsigned block_per_column = (height + blockDim.x - 1) / blockDim.x;
    const unsigned row_idx = (blockIdx.x % block_per_column) * blockDim.x + threadIdx.x;
    const unsigned col_idx = blockIdx.x / block_per_column;

    if(row_idx < height) {
        matrix_output[col_idx * height + row_idx] = matrix_input[row_idx * width + col_idx];
    }
}

// 读取合并
__global__ void transpose_naive_1D_v2(float* matrix_input, float* matrix_output, const unsigned height, const unsigned width){
    const unsigned block_per_row = (width + blockDim.x - 1) / blockDim.x;
    const unsigned row_idx = blockIdx.x / block_per_row;
    const unsigned col_idx = (blockIdx.x % block_per_row) * blockDim.x + threadIdx.x;

    if(col_idx < width) {
        matrix_output[col_idx * height + row_idx] = matrix_input[row_idx * width + col_idx];
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

#define TILE_SIZE 16
__global__ void transpose_shared(const float* matrix_input, float* matrix_output, const unsigned height, unsigned width) {
    __shared__ float tile[TILE_SIZE + 1][TILE_SIZE + 1];
    unsigned row_idx = TILE_SIZE * blockIdx.y + threadIdx.y;
    unsigned col_idx = TILE_SIZE * blockIdx.x + threadIdx.x;

    if(col_idx < width && row_idx < height) {
        tile[threadIdx.y][threadIdx.x] = matrix_input[row_idx * width + col_idx];
    }
    __syncthreads();

    row_idx = TILE_SIZE * blockIdx.x + threadIdx.y;
    col_idx = TILE_SIZE * blockIdx.y + threadIdx.x;
    if(col_idx < height && row_idx < width) {
        matrix_output[row_idx * height + col_idx] = tile[threadIdx.x][threadIdx.y];
    }

}

#define TILE_DIM 32
__global__ void transpose_shared_32(const float* matrix_input, float* matrix_output, const unsigned height, const unsigned width) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    unsigned row_idx = blockIdx.y * TILE_DIM + threadIdx.y;
    unsigned col_idx = blockIdx.x * TILE_DIM + threadIdx.x;

    #pragma unroll
    for(int i = 0; i < TILE_DIM; i+=8){
        if(col_idx < width && row_idx + i < height){
            tile[threadIdx.y + i][threadIdx.x] = matrix_input[(row_idx + i) * width + col_idx];
        }
    }
    __syncthreads();

    row_idx = blockIdx.x * TILE_DIM + threadIdx.y;
    col_idx = blockIdx.y * TILE_DIM + threadIdx.x;

    #pragma unroll
    for(int i = 0; i < TILE_DIM; i+=8){
        if(col_idx < height && row_idx + i < width){
            matrix_output[(row_idx + i) * height + col_idx] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

__global__ void transpose_shared_32_x4(float* matrix_input, float* matrix_output, const unsigned height, const unsigned width) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    unsigned row_idx = blockIdx.y * TILE_DIM + threadIdx.y;
    unsigned col_idx = blockIdx.x * TILE_DIM + 4 * threadIdx.x;

    if(col_idx < width && row_idx < height) {
        float4 reg_input = reinterpret_cast<float4*>(&matrix_input[row_idx * width + col_idx])[0];
        tile[threadIdx.y][4 * threadIdx.x] = reg_input.x;
        if(col_idx + 1 < width) tile[threadIdx.y][4 * threadIdx.x + 1] = reg_input.y;
        if(col_idx + 2 < width) tile[threadIdx.y][4 * threadIdx.x + 2] = reg_input.z;
        if(col_idx + 3 < width) tile[threadIdx.y][4 * threadIdx.x + 3] = reg_input.w;
    }

    __syncthreads();
    
    const unsigned new_row = threadIdx.y / 4;
    const unsigned new_col = threadIdx.x + (threadIdx.y % 4) * 8;
    row_idx = blockIdx.x * TILE_DIM + new_row;
    col_idx = blockIdx.y * TILE_DIM + new_col;

    for(int i = 0; i < TILE_DIM; i+=8){
        if(row_idx + i < width && col_idx < height) {
            matrix_output[(row_idx + i) * height + col_idx] = tile[new_col][new_row + i];
        }
    }
}

__global__ void transpose_shared_32_async(const float* matrix_input, float* matrix_output, const unsigned height, const unsigned width) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    unsigned row_idx = blockIdx.y * TILE_DIM + threadIdx.y;
    unsigned col_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    #pragma unroll
    for(int i = 0; i < TILE_DIM; i+=8){
        if(col_idx < width && row_idx + i < height){
            cg::memcpy_async(
                tile32, 
                tile[tile32.meta_group_rank() + i], 
                &matrix_input[(blockIdx.y * TILE_DIM + tile32.meta_group_rank() + i) * width + blockIdx.x * TILE_DIM], 
                sizeof(float) * tile32.size()
            );
        }
    }

    row_idx = blockIdx.x * TILE_DIM + threadIdx.y;
    col_idx = blockIdx.y * TILE_DIM + threadIdx.x;
    cg::wait(block);

    #pragma unroll
    for(int i = 0; i < TILE_DIM; i+=8){
        if(col_idx < height && row_idx + i < width){
            matrix_output[(row_idx + i) * height + col_idx] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

__global__ void transpose_shared_64(const float* matrix_input, float* matrix_output, const unsigned height, const unsigned width) {
    __shared__ float tile[64][65];

    const unsigned row_start = blockIdx.y * 64;
    const unsigned col_start = blockIdx.x * 64;
    unsigned row_idx = row_start + threadIdx.y;
    unsigned col_idx = col_start + threadIdx.x;

    #pragma unroll
    for(int i = 0; i < 64; i+=8){
        if(col_idx < width && row_idx + i < height){
            tile[threadIdx.y + i][threadIdx.x] = matrix_input[(row_idx + i) * width + col_idx];
            if(col_idx + 32 < width) tile[threadIdx.y + i][threadIdx.x + 32] = matrix_input[(row_idx + i) * width + col_idx + 32];
        }
    }
    __syncthreads();

    row_idx = col_start + threadIdx.y;
    col_idx = row_start + threadIdx.x;

    #pragma unroll
    for(int i = 0; i < 64; i+=8){
        if(col_idx < height && row_idx + i < width){
            matrix_output[(row_idx + i) * height + col_idx] = tile[threadIdx.x][threadIdx.y + i];
            if(col_idx + 32 < height) matrix_output[(row_idx + i) * height + col_idx + 32] = tile[threadIdx.x + 32][threadIdx.y + i];
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
    int *error_count;
    const unsigned row = 1<<15, col = 1<<15;
    CUDA_CHECK(cudaMalloc(&a, row * col * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, row * col * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&error_count, sizeof(int)));
    Timer* timer = new Timer();
    double duration = 0.0;
    
    // ---------------------------------
    // CUBLAS
    // ---------------------------------
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                row, col, &alpha, a, col,
                &beta, NULL, row, b, row
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        cublasDestroy(handle);
        std::cout << "cuBLAS: " << duration / max_run << " ms" << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // naive
    // ---------------------------------
    {
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((col + 15)/16, (row+15)/16);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_naive<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "naive: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // naivev2
    // ---------------------------------
    {
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((row + 15)/16, (col+15)/16);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_naive_v2<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "naive v2: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // naive 64
    // ---------------------------------
    {
        dim3 threadsPerBlock(32, 8);
        dim3 blocksPerGrid((col + 63)/64, (row+63)/64);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_naive_64<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "naive 64: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // naive 1D
    // ---------------------------------
    {
        unsigned threadsPerBlock = 256;
        unsigned blocksPerGrid = (row + threadsPerBlock - 1) / threadsPerBlock * col;
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_naive_1D<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "naive 1D: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // naive 1D v2
    // ---------------------------------
    {
        unsigned threadsPerBlock = 256;
        unsigned blocksPerGrid = (col + threadsPerBlock - 1) / threadsPerBlock * row;
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_naive_1D_v2<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "naive 1D v2: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // 4x
    // ---------------------------------
    {
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((row + 15)/16, (col/4 + 15)/16);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_x4<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "x4: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // shared
    // ---------------------------------
    {
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 blocksPerGrid((col + TILE_SIZE - 1)/TILE_SIZE, (row + TILE_SIZE - 1)/TILE_SIZE);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_shared<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "shared: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // shared opt 32
    // ---------------------------------
    {
        dim3 threadsPerBlock(32, 8);
        dim3 blocksPerGrid((col + 32 - 1)/32, (row + 32 - 1)/32);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_shared_32<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "shared opt: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // shared opt x4
    // ---------------------------------
    {
        dim3 threadsPerBlock(8, 32);
        dim3 blocksPerGrid((col + 32 - 1)/32, (row + 32 - 1)/32);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_shared_32_x4<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "shared opt x4: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // shared opt async
    // ---------------------------------
    {
        dim3 threadsPerBlock(32, 8);
        dim3 blocksPerGrid((col + 32 - 1)/32, (row + 32 - 1)/32);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_shared_32_async<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "shared opt async: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // shared opt 64
    // ---------------------------------
    {
        dim3 threadsPerBlock(32, 8);
        dim3 blocksPerGrid((col + 64 - 1)/64, (row + 64 - 1)/64);
        for(int i = 0; i < max_run; i++) {
            generateRandomData(a, row * col);
            timer->start();
            transpose_shared_64<<<blocksPerGrid, threadsPerBlock>>>(a, b, row, col);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
            verify_transpose(a, b, row, col, error_count);
            CUDA_CHECK(cudaMemset(b, 0, row * col * sizeof(float)));
        }
        std::cout << "shared opt 64: " << duration/max_run << std::endl;
        duration = 0.0;
    }

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
}