#include "timer.h"
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define FLOAT2(value) (reinterpret_cast<float2*>(&(value))[0])
#define CONST_FLOAT2(value) (reinterpret_cast<const float2*>(&(value))[0])

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
    const int tile_size = 32;
    __shared__ float sub_a[tile_size][tile_size];
    __shared__ float sub_b[tile_size][tile_size];
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int start_row_a = blockIdx.x * tile_size;
    const int start_col_b = blockIdx.y * tile_size;

    const unsigned elementsPerThread = 4; // tile_size * tile_size / block.num_threads()
    float output[elementsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};

    for(int start_col_a_row_b = 0; start_col_a_row_b < k; start_col_a_row_b += tile_size) {
        // 读入共享内存
        for(int i = warp.meta_group_rank(); i < tile_size; i += warp.meta_group_size()) {
           sub_a[i][warp.thread_rank()] = a[(start_col_a_row_b + i)* lda + start_row_a + warp.thread_rank()];
        }
        for(int i = warp.meta_group_rank(); i < tile_size; i += warp.meta_group_size()) {
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
    }
    for(int i = 0; i < 4; i++) {
        c[(start_col_b + warp.meta_group_rank() + i * warp.meta_group_size())* ldc + start_row_a + warp.thread_rank()] = output[i];
    }
}

// 相比之前的版本，thread处理的是有间隔的四个元素，这次处理一个2*2的tile
__global__ void sgemm_thread_tile(
    const int m, const int n, const int k, 
    const float* a, const int lda, 
    const float* b, const int ldb, 
    float *c, const int ldc
){
    const int tile_size = 32;
    __shared__ float sub_a[tile_size][tile_size + 1];
    __shared__ float sub_b[tile_size][tile_size + 1];
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int start_row_a = blockIdx.x * tile_size;
    const int start_col_b = blockIdx.y * tile_size;

    float output_tile[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    const unsigned tile_start_row = 2 * (block.thread_rank() / 16);
    const unsigned tile_start_col = 2 * (block.thread_rank() % 16);

    for(int start_col_a_row_b = 0; start_col_a_row_b < k; start_col_a_row_b += tile_size) {
        // 读入共享内存
        const float* a_start = a + start_col_a_row_b * lda + start_row_a;
        const float* b_start = b + start_col_b * ldb + start_col_a_row_b;
        for(int i = warp.meta_group_rank(); i < tile_size; i += warp.meta_group_size()) {
           sub_a[i][warp.thread_rank()] = a_start[i * lda  + warp.thread_rank()];
        }
        for(int i = warp.meta_group_rank(); i < tile_size; i += warp.meta_group_size()) {
           sub_b[i][warp.thread_rank()] = b_start[i * ldb + warp.thread_rank()];
        }
        __syncthreads();
        // 计算
        // 每个thread负责计算一个2*2的tile
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 32; j++) {
                float b_val = sub_b[tile_start_col + i][j];
                output_tile[i][0] += sub_a[j][tile_start_row] * b_val;
                output_tile[i][1] += sub_a[j][tile_start_row + 1] * b_val;
            }
        }
    }
    // 写回, 用sub_a中转一下，消除非合并写
    sub_a[tile_start_col][tile_start_row] = output_tile[0][0];
    sub_a[tile_start_col][tile_start_row + 1] = output_tile[0][1];
    sub_a[tile_start_col + 1][tile_start_row] = output_tile[1][0];
    sub_a[tile_start_col + 1][tile_start_row + 1] = output_tile[1][1];
    __syncthreads();
    for(int i = warp.meta_group_rank(); i < tile_size; i += warp.meta_group_size()){
        c[(start_col_b + i) * ldc + start_row_a + warp.thread_rank()] = sub_a[i][warp.thread_rank()];
    }
}

// 让一个thread负责处理连续的线程
__global__ void sgemm_thread_vec_tile(
    const int m, const int n, const int k, 
    const float* a, const int lda, 
    const float* b, const int ldb, 
    float *c, const int ldc
){
    const int tile_size = 32;
    __shared__ float sub_a[tile_size][tile_size + 1];
    __shared__ float sub_b[tile_size][tile_size + 1];
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const int start_row_a = blockIdx.x * tile_size;
    const int start_col_b = blockIdx.y * tile_size;

    float4 output_tile = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned tile_start_row = 4 * (block.thread_rank() % 8);
    const unsigned tile_start_col = block.thread_rank() / 8;

    for(int start_col_a_row_b = 0; start_col_a_row_b < k; start_col_a_row_b += tile_size) {
        // 读入共享内存
        const float* a_start = a + start_col_a_row_b * lda + start_row_a;
        const float* b_start = b + start_col_b * ldb + start_col_a_row_b;
        for(int i = warp.meta_group_rank(); i < tile_size; i += warp.meta_group_size()) {
           sub_a[i][warp.thread_rank()] = a_start[i * lda  + warp.thread_rank()];
        }
        for(int i = warp.meta_group_rank(); i < tile_size; i += warp.meta_group_size()) {
           sub_b[i][warp.thread_rank()] = b_start[i * ldb + warp.thread_rank()];
        }
        __syncthreads();
        // 计算
        // 每个thread负责计算1*4
        for(int i = 0; i < 32; i++) {
            float b_val = sub_b[tile_start_col][i];
            output_tile.x += sub_a[i][tile_start_row] * b_val;
            output_tile.y += sub_a[i][tile_start_row + 1] * b_val;
            output_tile.z += sub_a[i][tile_start_row + 2] * b_val;
            output_tile.w += sub_a[i][tile_start_row + 3] * b_val;
        }
        
    }
    // 写回
    float* c_start = c + start_col_b * ldc + start_row_a;
    *(reinterpret_cast<float4*>(&c_start[tile_start_col * ldc + tile_start_row])) = output_tile;
}

__global__ void sgemm_64(
    const int m, const int n, const int k, 
    const float* a, const int lda, 
    const float* b, const int ldb, 
    float *c, const int ldc
){
    const unsigned BLOCK_TILE_SIZE = 64;
    const unsigned THREAD_TILE_SIZE = 4;
    __shared__ float block_tile_a[BLOCK_TILE_SIZE][BLOCK_TILE_SIZE];
    __shared__ float block_tile_b[BLOCK_TILE_SIZE][BLOCK_TILE_SIZE+1];

    auto block = cg::this_thread_block();
    auto tile_64 = cg::tiled_partition<BLOCK_TILE_SIZE>(block);
    auto warp = cg::tiled_partition<32>(block);

    const float* a_ptr = a + blockIdx.x * BLOCK_TILE_SIZE;
    const float* b_ptr = b + blockIdx.y * BLOCK_TILE_SIZE * ldb;
    float* c_ptr = c + blockIdx.y * BLOCK_TILE_SIZE * ldb + blockIdx.x * BLOCK_TILE_SIZE;

    float4 thread_tile_c[THREAD_TILE_SIZE] ={0};
    const unsigned thraed_tile_row = THREAD_TILE_SIZE * (block.thread_rank() / (BLOCK_TILE_SIZE/THREAD_TILE_SIZE));
    const unsigned thraed_tile_col = THREAD_TILE_SIZE * (block.thread_rank() % (BLOCK_TILE_SIZE/THREAD_TILE_SIZE));

    for(int start_k = 0; start_k < k; start_k += BLOCK_TILE_SIZE){
        const float* a_tile_start = a_ptr + start_k * lda;
        const float* b_tile_start = b_ptr + start_k;

        // 读共享内存
        #pragma unroll
        for(int i = tile_64.meta_group_rank(); i < BLOCK_TILE_SIZE; i+=tile_64.meta_group_size()) {
            block_tile_a[i][tile_64.thread_rank()] = a_tile_start[i * lda + tile_64.thread_rank()];
        }
        #pragma unroll
        for(int i = tile_64.meta_group_rank(); i < BLOCK_TILE_SIZE; i+=tile_64.meta_group_size()) {
            block_tile_b[i][tile_64.thread_rank()] = b_tile_start[i * ldb + tile_64.thread_rank()];
        }
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < THREAD_TILE_SIZE; i++ ){
            for(int j = 0; j < BLOCK_TILE_SIZE; j++){
                float b_val = block_tile_b[thraed_tile_col + i][j];
                thread_tile_c[i].x += block_tile_a[j][thraed_tile_row] * b_val;
                thread_tile_c[i].y += block_tile_a[j][thraed_tile_row + 1] * b_val;
                thread_tile_c[i].z += block_tile_a[j][thraed_tile_row + 2] * b_val;
                thread_tile_c[i].w += block_tile_a[j][thraed_tile_row + 3] * b_val;
            }
        }
    }

    // 写回
    FLOAT4(c_ptr[ldc * thraed_tile_col + thraed_tile_row]) = thread_tile_c[0];
    FLOAT4(c_ptr[ldc * (thraed_tile_col + 1) + thraed_tile_row]) = thread_tile_c[1];
    FLOAT4(c_ptr[ldc * (thraed_tile_col + 2) + thraed_tile_row]) = thread_tile_c[2];
    FLOAT4(c_ptr[ldc * (thraed_tile_col + 3) + thraed_tile_row]) = thread_tile_c[3];
}

__global__ void sgemm_64_opt(
    const int m, const int n, const int k, 
    const float* a, const int lda, 
    const float* b, const int ldb, 
    float *c, const int ldc
){
    const unsigned BLOCK_TILE_SIZE = 64;
    const unsigned THREAD_TILE_SIZE = 4;
    __shared__ float block_tile_a[BLOCK_TILE_SIZE][BLOCK_TILE_SIZE];
    __shared__ float block_tile_b[BLOCK_TILE_SIZE][BLOCK_TILE_SIZE+1];

    auto block = cg::this_thread_block();
    auto tile_64 = cg::tiled_partition<BLOCK_TILE_SIZE>(block);
    auto tile_16 = cg::tiled_partition<16>(block);
    auto warp = cg::tiled_partition<32>(block);

    const float* a_ptr = a + blockIdx.x * BLOCK_TILE_SIZE;
    const float* b_ptr = b + blockIdx.y * BLOCK_TILE_SIZE * ldb;
    float* c_ptr = c + blockIdx.y * BLOCK_TILE_SIZE * ldb + blockIdx.x * BLOCK_TILE_SIZE;

    float4 thread_tile_c[THREAD_TILE_SIZE] ={0};
    const unsigned thraed_tile_row = THREAD_TILE_SIZE * (block.thread_rank() / (BLOCK_TILE_SIZE/THREAD_TILE_SIZE));
    const unsigned thraed_tile_col = THREAD_TILE_SIZE * (block.thread_rank() % (BLOCK_TILE_SIZE/THREAD_TILE_SIZE));

    for(int start_k = 0; start_k < k; start_k += BLOCK_TILE_SIZE){
        const float* a_tile_start = a_ptr + start_k * lda;
        const float* b_tile_start = b_ptr + start_k;

        // 读取全局内存，写入共享内存，全局内存的列放在共享内存的行
        for(int i = tile_16.meta_group_rank(); i < BLOCK_TILE_SIZE; i+=tile_16.meta_group_size()) {
            float4 reg_a = CONST_FLOAT4(a_tile_start[i * lda + 4 * tile_16.thread_rank()]);
            block_tile_a[i][4 * tile_16.thread_rank()] = reg_a.x;
            block_tile_a[i][4 * tile_16.thread_rank() + 1] = reg_a.y;
            block_tile_a[i][4 * tile_16.thread_rank() + 2] = reg_a.z;
            block_tile_a[i][4 * tile_16.thread_rank() + 3] = reg_a.w;
        }
        // 转置读入共享内存
        #pragma unroll
        for(int i = tile_64.meta_group_rank(); i < BLOCK_TILE_SIZE; i+=tile_64.meta_group_size()) {
            block_tile_b[tile_64.thread_rank()][i] = b_tile_start[i * ldb + tile_64.thread_rank()];
        }
        __syncthreads();
        float reg_a[THREAD_TILE_SIZE];
        float reg_b[THREAD_TILE_SIZE];
        #pragma unroll
        for (int j = 0; j < BLOCK_TILE_SIZE; j++) {
            reg_a[0] = block_tile_a[j][thraed_tile_row + 0];
            reg_a[1] = block_tile_a[j][thraed_tile_row + 1];
            reg_a[2] = block_tile_a[j][thraed_tile_row + 2];
            reg_a[3] = block_tile_a[j][thraed_tile_row + 3];
            reg_b[0] = block_tile_b[j][thraed_tile_col + 0];
            reg_b[1] = block_tile_b[j][thraed_tile_col + 1];
            reg_b[2] = block_tile_b[j][thraed_tile_col + 2];
            reg_b[3] = block_tile_b[j][thraed_tile_col + 3];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_SIZE; i++) {
                thread_tile_c[i].x += reg_a[0] * reg_b[i];
                thread_tile_c[i].y += reg_a[1] * reg_b[i];
                thread_tile_c[i].z += reg_a[2] * reg_b[i];
                thread_tile_c[i].w += reg_a[3] * reg_b[i];
            }
        }
    }

    // 写回
    FLOAT4(c_ptr[ldc * thraed_tile_col + thraed_tile_row]) = thread_tile_c[0];
    FLOAT4(c_ptr[ldc * (thraed_tile_col + 1) + thraed_tile_row]) = thread_tile_c[1];
    FLOAT4(c_ptr[ldc * (thraed_tile_col + 2) + thraed_tile_row]) = thread_tile_c[2];
    FLOAT4(c_ptr[ldc * (thraed_tile_col + 3) + thraed_tile_row]) = thread_tile_c[3];
}

__global__ void sgemm_128(
    const int m, const int n, const int k, 
    const float* a, const int lda, 
    const float* b, const int ldb, 
    float *c, const int ldc
){
    const unsigned BLOCK_TILE_SIZE = 128;
    __shared__ float block_tile_a[8][BLOCK_TILE_SIZE];
    __shared__ float block_tile_b[8][BLOCK_TILE_SIZE+4];

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    auto tile_8 = cg::tiled_partition<8>(block);

    const float* a_ptr = a + blockIdx.x * BLOCK_TILE_SIZE;
    const float* b_ptr = b + blockIdx.y * BLOCK_TILE_SIZE * ldb;
    float* c_ptr = c + blockIdx.y * BLOCK_TILE_SIZE * ldb + blockIdx.x * BLOCK_TILE_SIZE;

    // thread tile为4*16
    float4 thread_tile_c[16] ={0};
    const unsigned thraed_tile_row = 4 * warp.thread_rank();
    const unsigned thraed_tile_col = 16 * warp.meta_group_rank();

    for(int start_k = 0; start_k < k; start_k += 8){
        const float* a_tile_start = a_ptr + start_k * lda;
        const float* b_tile_start = b_ptr + start_k;

        // 读取a tile
        (reinterpret_cast<float4*>(block_tile_a[warp.meta_group_rank()]))[warp.thread_rank()] = CONST_FLOAT4(a_tile_start[warp.meta_group_rank() * lda + 4 * warp.thread_rank()]);
        // 读取b tile
        #pragma unroll
        for(int i = tile_8.meta_group_rank(); i < BLOCK_TILE_SIZE; i+=tile_8.meta_group_size()) {
            // 此处有bank conflict
            block_tile_b[tile_8.thread_rank()][i] = b_tile_start[i * ldb + tile_8.thread_rank()];
        }
        __syncthreads();

        // 计算
        float reg_a[4];
        float reg_b[16];
        
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            #pragma unroll
            for(int i = 0; i < 4; i++){
                reg_a[i] = block_tile_a[j][thraed_tile_row + i];
            }
            #pragma unroll
            for(int i = 0; i < 16; i++){
                reg_b[i] = block_tile_b[j][thraed_tile_col + i];
                thread_tile_c[i].x += reg_a[0] * reg_b[i];
                thread_tile_c[i].y += reg_a[1] * reg_b[i];
                thread_tile_c[i].z += reg_a[2] * reg_b[i];
                thread_tile_c[i].w += reg_a[3] * reg_b[i];
            }
        }
    }
    // 写回
    #pragma unroll
    for(int i = 0; i < 16; i++){
        FLOAT4(c_ptr[ldc * (thraed_tile_col + i) + thraed_tile_row]) = thread_tile_c[i];
    }
}

__global__ void sgemm_128_pipeline(
    const int m, const int n, const int k,
    const float* a, const int lda,
    const float* b, const int ldb,
    float* c, const int ldc
) {
    const unsigned BLOCK_TILE_SIZE = 128;
    // 1. 双缓冲：为A和B的tile创建两个缓冲区
    __shared__ float block_tile_a[2][8][BLOCK_TILE_SIZE];
    __shared__ float block_tile_b[2][8][BLOCK_TILE_SIZE + 4]; // Padding保持，以减少bank conflict

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    auto tile_8 = cg::tiled_partition<8>(block);

    const float* a_ptr = a + blockIdx.x * BLOCK_TILE_SIZE;
    const float* b_ptr = b + blockIdx.y * BLOCK_TILE_SIZE * ldb;
    float* c_ptr = c + blockIdx.y * BLOCK_TILE_SIZE * ldb + blockIdx.x * BLOCK_TILE_SIZE;

    float4 thread_tile_c[16] = {0};
    const unsigned thread_tile_row = 4 * warp.thread_rank();
    const unsigned thread_tile_col = 16 * warp.meta_group_rank();

    // --- 流水线启动 (Prologue) ---
    // 预取第一个k-tile (k=0) 的数据到缓冲区的 [0]
    const float* a_tile_prologue = a_ptr; // start_k = 0
    const float* b_tile_prologue = b_ptr; // start_k = 0

    // 读取 a tile (k=0)
    (reinterpret_cast<float4*>(block_tile_a[0][warp.meta_group_rank()]))[warp.thread_rank()] = 
        *reinterpret_cast<const float4*>(&a_tile_prologue[warp.meta_group_rank() * lda + 4 * warp.thread_rank()]);

    // 读取 b tile (k=0)
    #pragma unroll
    for (int i = tile_8.meta_group_rank(); i < BLOCK_TILE_SIZE; i += tile_8.meta_group_size()) {
        block_tile_b[0][tile_8.thread_rank()][i] = b_tile_prologue[i * ldb + tile_8.thread_rank()];
    }
    __syncthreads();

    // --- 流水线主体 (Main Loop) ---
    int pingpong_idx = 0;
    for (int start_k = 8; start_k < k; start_k += 8) {
        // 预取下一个k-tile的数据到另一个缓冲区
        const float* a_tile_prefetch = a_ptr + start_k * lda;
        const float* b_tile_prefetch = b_ptr + start_k;
        
        // 异步加载下一个A tile到 `1-pingpong_idx` 缓冲区
        (reinterpret_cast<float4*>(block_tile_a[1 - pingpong_idx][warp.meta_group_rank()]))[warp.thread_rank()] = 
            *reinterpret_cast<const float4*>(&a_tile_prefetch[warp.meta_group_rank() * lda + 4 * warp.thread_rank()]);

        // 异步加载下一个B tile到 `1-pingpong_idx` 缓冲区
        #pragma unroll
        for (int i = tile_8.meta_group_rank(); i < BLOCK_TILE_SIZE; i += tile_8.meta_group_size()) {
            block_tile_b[1 - pingpong_idx][tile_8.thread_rank()][i] = b_tile_prefetch[i * ldb + tile_8.thread_rank()];
        }

        // --- 计算当前k-tile ---
        // 使用 `pingpong_idx` 缓冲区的数据进行计算
        float reg_a[4];
        float reg_b[16];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                reg_a[i] = block_tile_a[pingpong_idx][j][thread_tile_row + i];
            }
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                reg_b[i] = block_tile_b[pingpong_idx][j][thread_tile_col + i];
                thread_tile_c[i].x += reg_a[0] * reg_b[i];
                thread_tile_c[i].y += reg_a[1] * reg_b[i];
                thread_tile_c[i].z += reg_a[2] * reg_b[i];
                thread_tile_c[i].w += reg_a[3] * reg_b[i];
            }
        }
        
        // 同步，确保预取完成，并切换缓冲区
        __syncthreads();
        pingpong_idx = 1 - pingpong_idx;
    }

    // --- 流水线收尾 (Epilogue) ---
    // 计算最后一个k-tile的数据（这些数据在循环的最后一次迭代中被加载）
    float reg_a[4];
    float reg_b[16];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            reg_a[i] = block_tile_a[pingpong_idx][j][thread_tile_row + i];
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            reg_b[i] = block_tile_b[pingpong_idx][j][thread_tile_col + i];
            thread_tile_c[i].x += reg_a[0] * reg_b[i];
            thread_tile_c[i].y += reg_a[1] * reg_b[i];
            thread_tile_c[i].z += reg_a[2] * reg_b[i];
            thread_tile_c[i].w += reg_a[3] * reg_b[i];
        }
    }

    // 写回
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        *reinterpret_cast<float4*>(&c_ptr[ldc * (thread_tile_col + i) + thread_tile_row]) = thread_tile_c[i];
    }
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "input max run times!" << std::endl;
        return -1;
    }
    cudaSetDevice(1);
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

    // // ---------------------------------
    // // naive
    // // ---------------------------------
    // {
    //     const unsigned threadsPerBlock = 256;
    //     const dim3 blocksPerGrid(CEIL(m, 32), CEIL(n, 32));
    //     for(int i = 0; i < max_run; i++) {
    //         timer->start();
    //         sgemm_naive<<<blocksPerGrid, threadsPerBlock>>>(m,n,k,a,m,b,k,c,m);
    //         CUDA_CHECK(cudaGetLastError());
    //         CUDA_CHECK(cudaDeviceSynchronize());
    //         duration += timer->stop();
    //         verify_result(c, standard_output, m, n, m, m);
    //     }
    //     std::cout << "naive: " << duration / max_run << " ms" << std::endl;
    //     duration = 0.0;
    // }

    // // ---------------------------------
    // // shared
    // // ---------------------------------
    // {
    //     const unsigned threadsPerBlock = 256;
    //     const dim3 blocksPerGrid(CEIL(m, 32), CEIL(n, 32));
    //     for(int i = 0; i < max_run; i++) {
    //         timer->start();
    //         sgemm_shared<<<blocksPerGrid, threadsPerBlock>>>(m,n,k,a,m,b,k,c,m);
    //         CUDA_CHECK(cudaGetLastError());
    //         CUDA_CHECK(cudaDeviceSynchronize());
    //         duration += timer->stop();
    //         verify_result(c, standard_output, m, n, m, m);
    //     }
    //     std::cout << "shared: " << duration / max_run << " ms" << std::endl;
    //     duration = 0.0;
    // }

    // // ---------------------------------
    // // thread tile
    // // ---------------------------------
    // {
    //     const unsigned threadsPerBlock = 256;
    //     const dim3 blocksPerGrid(CEIL(m, 32), CEIL(n, 32));
    //     for(int i = 0; i < max_run; i++) {
    //         timer->start();
    //         sgemm_thread_tile<<<blocksPerGrid, threadsPerBlock>>>(m,n,k,a,m,b,k,c,m);
    //         CUDA_CHECK(cudaGetLastError());
    //         CUDA_CHECK(cudaDeviceSynchronize());
    //         duration += timer->stop();
    //         verify_result(c, standard_output, m, n, m, m);
    //     }
    //     std::cout << "thread tile: " << duration / max_run << " ms" << std::endl;
    //     duration = 0.0;
    // }

    // // ---------------------------------
    // // vec thread tile
    // // ---------------------------------
    // {
    //     const unsigned threadsPerBlock = 256;
    //     const dim3 blocksPerGrid(CEIL(m, 32), CEIL(n, 32));
    //     for(int i = 0; i < max_run; i++) {
    //         timer->start();
    //         sgemm_thread_vec_tile<<<blocksPerGrid, threadsPerBlock>>>(m,n,k,a,m,b,k,c,m);
    //         CUDA_CHECK(cudaGetLastError());
    //         CUDA_CHECK(cudaDeviceSynchronize());
    //         duration += timer->stop();
    //         verify_result(c, standard_output, m, n, m, m);
    //     }
    //     std::cout << "vector tile: " << duration / max_run << " ms" << std::endl;
    //     duration = 0.0;
    // }

    // ---------------------------------
    // tile 64
    // ---------------------------------
    {
        const unsigned threadsPerBlock = 256;
        const dim3 blocksPerGrid(CEIL(m, 64), CEIL(n, 64));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            sgemm_64<<<blocksPerGrid, threadsPerBlock>>>(m,n,k,a,m,b,k,c,m);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            //verify_result(c, standard_output, m, n, m, m);
        }
        std::cout << "tile 64: " << duration / max_run << " ms" << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // tile 64 opt
    // ---------------------------------
    {
        const unsigned threadsPerBlock = 256;
        const dim3 blocksPerGrid(CEIL(m, 64), CEIL(n, 64));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            sgemm_64_opt<<<blocksPerGrid, threadsPerBlock>>>(m,n,k,a,m,b,k,c,m);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            //verify_result(c, standard_output, m, n, m, m);
        }
        std::cout << "tile 64 opt: " << duration / max_run << " ms" << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // tile 128
    // ---------------------------------
    {
        const unsigned threadsPerBlock = 256;
        const dim3 blocksPerGrid(CEIL(m, 128), CEIL(n, 128));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            sgemm_128<<<blocksPerGrid, threadsPerBlock>>>(m,n,k,a,m,b,k,c,m);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            //verify_result(c, standard_output, m, n, m, m);
        }
        std::cout << "tile 128: " << duration / max_run << " ms" << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // tile 128 pipeline
    // ---------------------------------
    {
        const unsigned threadsPerBlock = 256;
        const dim3 blocksPerGrid(CEIL(m, 128), CEIL(n, 128));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            sgemm_128_pipeline<<<blocksPerGrid, threadsPerBlock>>>(m,n,k,a,m,b,k,c,m);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            //verify_result(c, standard_output, m, n, m, m);
        }
        std::cout << "tile 128: " << duration / max_run << " ms" << std::endl;
        duration = 0.0;
    }
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(standard_output));
}