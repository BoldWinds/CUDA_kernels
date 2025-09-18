# gemm

通用矩阵乘法, 以下计算默认为列优先

## sgemm naive

```cuda
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
```

在这个版本中我使用了不少的优化方法。首先，每个block有256线程，处理32*32的结果矩阵子块；对于a和b的32\*k的块，每次读入32\*32大小，迭代计算。

这个版本的性能做到了cuBLAS的八分之一，性能相当差。主要问题在于，cuBLAS首先是没有Global到L1 Cache的访问的；并且在总读取量上，无论是读取Device Memory还是访问Shared Memory，我的访问量都是cuBLAS的整整20倍，说明很多可以省去的计算我没有做到

## sgemm opt

