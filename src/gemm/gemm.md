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

除此以外, 我还探究了一下不使用Shared Memory做中转来实现合并访问方法: 让一个线程处理全局内存中连续的四个元素, 写入的时候直接float4向量化写入即可.
但是这样做并没有提升性能, 是因为MIO Throttle Stall提升了: 一个thread原本需要访问两行两列, 变成了四行一列, 对shared memory的压力增加了25%; 但是优化后省去的写回前shared memory中转都不会造成这么多的MIO Throttle Stall, 所以性能变差了

## 64 tile

之前有提到, 我的版本总读取量更高; 后面经过分析, 不难看出, 总读取量是和tile size有关系的, 对于$size*size$大小的tile, 每一个block要读取$size*k$长度的a和$k*size$长度的b, 一共有$m*n/size*size$个block, 那么总读取量就是$m*n*k*2/size$. 所以解决这个问题的好方案就是增大tile, 先增大到$64*64$看看

```cuda
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

    float4 *shared_prt = (float4*)block_tile_a;
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
```

果然，更新这个版本的代码之后，总的读取量仅为cuBLAS版本的1.6倍；性能也大幅提高，达到cuBLAS的30%

继续分析性能问题，发现读取全局内存到共享内存时，存在大量的stall，约占stall总数的一半；观察发现这时的写入一次仅写入32位，看看能否向量化写入，一次操作128位；除此以外，部分shared load指令出现了L1 Wavefronts Shared Excessive，这一般是因为bank conflict，并且出现该问题的指令都不是128位的，别的没有这问题的指令都是128位的

那么下一步的优化就主要是对共享内存读写的优化了，也应当包括向量化的读取/写入

## 64 tile opt

```cuda
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
```

优化了一下全局内存的读取，降低了bank conflict，达到cuBLAS 50%的性能

## 128 tile
到了这个tile大小，就得考虑一个问题，共享内存放不下这么大的tile了，所以要切更小的块：
- 对于a，一次性读128*8
- 对于b，一次性读8*128

```cuda
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
```

如果每个thread的tile选择$8*8$的话，那么在写回时会产生大量的非合并访问，因此，每个thread的tile大小为$4*16$，保证全局内存的写是合并的

pipeline版本没有使用memcpy_async，使用的是算法上的流水线化，性能提升不大。

该版本达到了70%的性能，但是还是有距离

先这样吧，再改感觉也不知道该怎么改才好了，等后面有精力再把这一块好好写一下
