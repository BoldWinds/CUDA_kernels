#include "timer.h"
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void verify_result_kernel(
    const float* computed,
    const float* standard,
    int rows, int cols,
    int ld_computed,
    int ld_standard,
    int* error_count,
    float relative_tolerance = 1e-3f,
    float absolute_tolerance = 1e-5f)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int computed_idx = col * ld_computed + row;
        int standard_idx = col * ld_standard + row;

        float computed_val = computed[computed_idx];
        float standard_val = standard[standard_idx];
        float diff = fabsf(computed_val - standard_val);

        if (diff > (absolute_tolerance + relative_tolerance * fabsf(standard_val))) {
            atomicAdd(error_count, 1);
        }
    }
}

void verify_result(
    const float* computed,
    const float* standard,
    int rows, int cols,
    int ld_computed,
    int ld_standard,
    float relative_tolerance = 1e-3f,
    float absolute_tolerance = 1e-5f
){
    
    int *d_error_count;
    CUDA_CHECK(cudaMalloc(&d_error_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_error_count, 0, sizeof(int)));
    dim3 threadsPerBlock(16, 16);
    // 计算需要的block数量，确保能覆盖整个矩阵
    dim3 numBlocks( (rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (cols + threadsPerBlock.y - 1) / threadsPerBlock.y );
    verify_result_kernel<<<numBlocks, threadsPerBlock>>>(
        computed,
        standard,
        rows, cols,
        ld_computed, ld_standard,
        d_error_count
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int error_count = 0;
    CUDA_CHECK(cudaMemcpy(&error_count, d_error_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (error_count != 0) {
        std::cout << "ERROR: Verification failed! Found " << error_count << " mismatch(es)." << std::endl;
        std::cout << "       The provided matrix is NOT a correct transpose." << std::endl;
    }

}

// 256 threads/block, grid=(row/32,col/32), 对应c中的32*32的子块, a中的32*k, b中的k*32
// 对于a和b中的子块，太大了，每次处理32*32个，总共处理(k/32)+1次
__global__ void sgemm_naive(
    const int m, const int n, const int k, 
    const float* a, const int lda, 
    const float* b, const int ldb, 
    float *c, const int ldc
){
    const int output_row = 32, output_col = 32;
    __shared__ float sub_a[output_row][output_col + 1];
    __shared__ float sub_b[output_col][output_row];
    __shared__ float output[output_row][output_col+1];
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int start_row_a = blockIdx.x * output_row;
    const int start_col_b = blockIdx.y * output_col; 

    // output清零
    for(int i = warp.meta_group_rank(); i < output_row; i += warp.meta_group_size()){
        output[i][warp.thread_rank()] = 0.0f;
    }
    __syncthreads();

    for(int start_col_a_row_b = 0; start_col_a_row_b < k; start_col_a_row_b += output_col) {
        // 读入共享内存, 读取时对sub_a转置
        for(int i = warp.meta_group_rank(); i < output_col; i += warp.meta_group_size()) {
           sub_a[warp.thread_rank()][i] = a[(start_col_a_row_b + i)* lda + start_row_a + warp.thread_rank()];
        }
        for(int i = warp.meta_group_rank(); i < output_row; i += warp.meta_group_size()) {
           sub_b[i][warp.thread_rank()] = b[(start_col_b + i)* ldb + start_col_a_row_b + warp.thread_rank()];
        }
        __syncthreads();
        // 计算
        for(int i = warp.meta_group_rank(); i < output_row; i += warp.meta_group_size()){
            float a_val =  sub_a[i][warp.thread_rank()];
            float sum = a_val * sub_b[warp.thread_rank()][warp.thread_rank()];
            for(int j = 1; j < output_col; j++) {
                const unsigned src_lane = (warp.thread_rank() + j) % warp.num_threads();
                sum += warp.shfl(a_val, src_lane) * sub_b[warp.thread_rank()][src_lane];
            }
            output[i][warp.thread_rank()] += sum;
        }
    }
    __syncthreads();
    for(int i = warp.meta_group_rank(); i < output_col; i += warp.meta_group_size()) {
        c[(start_col_b + i)* ldc + start_row_a + warp.thread_rank()] = output[warp.thread_rank()][i];
    }
}

// 对共享内存的优化版本
__global__ void sgemm_shared(
    const int m, const int n, const int k, 
    const float* a, const int lda, 
    const float* b, const int ldb, 
    float *c, const int ldc
){
    const int tile_size = 32, output_row = 32, output_col = 32;
    __shared__ float sub_a[output_col][output_row];
    __shared__ float sub_b[output_col][output_row];
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int start_row_a = blockIdx.x * tile_size;
    const int start_col_b = blockIdx.y * tile_size;

    const unsigned elementsPerThread = 4; // tile_size * tile_size / block.num_threads()
    float output[elementsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};

    for(int start_col_a_row_b = 0; start_col_a_row_b < k; start_col_a_row_b += output_col) {
        // 读入共享内存
        for(int i = warp.meta_group_rank(); i < output_col; i += warp.meta_group_size()) {
           sub_a[i][warp.thread_rank()] = a[(start_col_a_row_b + i)* lda + start_row_a + warp.thread_rank()];
        }
        for(int i = warp.meta_group_rank(); i < output_row; i += warp.meta_group_size()) {
           sub_b[i][warp.thread_rank()] = b[(start_col_b + i)* ldb + start_col_a_row_b + warp.thread_rank()];
        }
        __syncthreads();
        // 计算
        // 每个warp负责整个sub_a乘sub_b中的1列，计算的结果对应结果tile的一列
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 32; j++){
                output[i] += sub_a[j][warp.thread_rank()] * sub_b[warp.meta_group_rank() + i * warp.meta_group_size()][j];
            }
        }
        __syncthreads(); 
    }
    for(int i = 0; i < 4; i++) {
        c[(start_col_b + warp.meta_group_rank() + i * warp.meta_group_size())* ldc + start_row_a + warp.thread_rank()] = output[i];
    }
}


int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "input max run times!" << std::endl;
        return -1;
    }
    int max_run = std::atoi(argv[1]);
    float *a, *b, *c, *standard_output;
    const unsigned m = 1<<12, n = 1<<12, k = 1<<12;
    CUDA_CHECK(cudaMalloc(&a, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&standard_output, m * n * sizeof(float)));
    generateRandomData(a, m * k);
    generateRandomData(b, k * n);
    CUDA_CHECK(cudaDeviceSynchronize());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CudaTimer* timer = new CudaTimer();
    double duration = 0.0;
    
    // ---------------------------------
    // CUBLAS
    // ---------------------------------
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        for(int i = 0; i < max_run; i++) {
            timer->start();
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                &alpha, a, m, b, k, &beta, standard_output,  m
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            
        }
        cublasDestroy(handle);
        std::cout << "cuBLAS: " << duration / max_run << " ms" << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // naive
    // ---------------------------------
    {
        const unsigned threadsPerBlock = 256;
        const dim3 blocksPerGrid(CEIL(m, 32), CEIL(n, 32));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            sgemm_naive<<<blocksPerGrid, threadsPerBlock>>>(m,n,k,a,m,b,k,c,m);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            verify_result(c, standard_output, m, n, m, m);
        }
        std::cout << "naive: " << duration / max_run << " ms" << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // shared
    // ---------------------------------
    {
        const unsigned threadsPerBlock = 256;
        const dim3 blocksPerGrid(CEIL(m, 32), CEIL(n, 32));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            sgemm_shared<<<blocksPerGrid, threadsPerBlock>>>(m,n,k,a,m,b,k,c,m);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            verify_result(c, standard_output, m, n, m, m);
        }
        std::cout << "shared: " << duration / max_run << " ms" << std::endl;
        duration = 0.0;
    }

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(standard_output));
}