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

这个版本的性能做到了cuBLAS的7%，性能相当差。主要问题在于，cuBLAS首先是没有Global到L1 Cache的访问的；并且在总读取量上，无论是读取Device Memory还是访问Shared Memory，我的访问量都是cuBLAS的整整20倍，说明很多可以省去的计算我没有做到；
除此以外，Warp Stall的原因中，Stall MIO Throttle、Stall Long Scoreboard、Stall Wait和Stall Short。想要进行优化，就要降低这几种Stall，而Stall MIO Throttle一般就是因为读写Shared Memory所引起的，其他的几种一般也是由于访存引起，适当添加每个warp的计算强度可以掩盖一定的stall

## sgemm opt

```cuda
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
```

优化后的版本，没有改变访问全局内存的方式，但是优化了共享内存的访问。把output的tile从共享内存中移除，转而使用寄存器存储，大大提高了性能，达到了cuBLAS的20%性能。并且，Stall MIO Throttle显著下降，虽然没有下降到与cuBLAS相同的等级，但也是有所提升。除此以外，对全局内存的读取方面没有大提升，仍然读取了很多的冗余数据。

PS:
```cuda
        float reg_a[32];
        for(int j = 0; j < 32; j++){
            reg_a[j] = sub_a[j][warp.thread_rank()];
            output[0] += reg_a[j] * sub_b[warp.meta_group_rank()][j];
        }
        for(int i = 1; i < 4; i++) {
            for(int j = 0; j < 32; j++){
                output[i] += reg_a[j] * sub_b[warp.meta_group_rank() + i * warp.meta_group_size()][j];
            }
        }
```
这是对于计算tile的矩阵乘法的代码的修改版本，乍一看他避免了重复读取sub_a，变为只读取一次，将其保存在寄存器里然后与sub_b中的元素相乘，应该有性能提升？但是实际上没有，和前面的版本在性能上以及寄存器使用上都没有区别，说明对于这种固定的循环，编译器完全可以分析其中哪些部分数据是重复使用的，并把它们放到寄存器中加速

## sgemm thread tile

```cuda
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
```

比起前面的版本，这次一个thread负责处理一个$2*2$的tile，block tile由256这样的小tile组成。这样做让每个thread读取sub_a中的两行和sub_b中的两列，计算得出四个元素，重用了数据。

分析显示，计算利用率达到92%，内存吞吐率达到92%，性能提升12%，达到cuBLAS的22%的性能

