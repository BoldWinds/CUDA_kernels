# transpose

## 数据

暂定为$2^{15}$行, $2^{15}$列

## CUBLAS

首先是cublas的对照组实现：
```cuda
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
```

kerel用时10.26ms，计算利用率34.3%，内存利用率91.9%

## naive

```cuda
// 写入是合并的
__global__ void transpose_naive(float* matrix_input, float* matrix_output, unsigned height, unsigned width){
    unsigned row_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned col_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if(col_idx < width && row_idx < height) {
        matrix_output[col_idx * height + row_idx] = matrix_input[row_idx * width + col_idx];
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
```
两个kernel最大的区别在于，前者优先保证合并写入，后者优先保证读取。
结果是前者用时15.03ms，计算利用率15.6%，内存利用率62.7%，全面低于CUBLAS。观察Grid Size，发现使用的线程数是cublas的16倍，所以应该尝试一个thread处理更多的数据，提高内存利用率。
而后者用时接近20ms，可以说在这个GPU上比起保证合并读，更应优先保证合并写。

```cuda
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
```

两个一维的版本性能更差，甚至差得多，这是因为一维的block，要么在读取时会产生cache thrashing，要么在写入时产生，造成额外读取/写入内存的量

这两个kernel告诉我们, 对于transpose这种二维属性的数据, 就要用二维的block去处理, 不然就会在读取或写入时产生大量内存的抖动, 严重影响性能

## x4

```cuda
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
```

这份代码在不借助任何shared memory的情况下，达到了cublas转置98%的性能，原因在于其内存吞吐率和cublas的几乎一致。
但是这份代码还存在一个问题：对内存的写入是非合并的

## shared

借助shared memory，尝试实现读取和写入的双合并

```cuda
#define TILE_SIZE 16

__global__ void transpose_shared(float* matrix_input, float* matrix_output, const unsigned height, unsigned width) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    unsigned row_idx = TILE_SIZE * blockIdx.y + threadIdx.y;
    unsigned col_idx = TILE_SIZE * blockIdx.x + threadIdx.x;

    if(col_idx < width && row_idx < height) {
        tile[threadIdx.y][threadIdx.x] = matrix_input[row_idx * width + col_idx];
    }
    __syncthreads();

    row_idx = TILE_SIZE * blockIdx.x + threadIdx.y;
    col_idx = TILE_SIZE * blockIdx.y + threadIdx.x;
    if(col_idx < width && row_idx < height) {
        matrix_output[row_idx * height + col_idx] = tile[threadIdx.x][threadIdx.y];
    }

}
```

这样做消除了全局内存的非合并访问,在nsight compute的分析结果中, 仍然性能不如native版本. 因为它虽然消除了全局内存的非合并访问, 但是引入了shared memory的非合并访问, 有一半对shared memory的访问都是非合并的, 该怎么解决呢? 

优化的办法就是降低bank conflict. 一般来说, 给shared memory添加padding就能解决了. 但是由于在这个kernel中, shared memory的大小为$16\*16$, 这导致即使添加了一个padding, 也会发生2-way bank conflict: 比如对于threadIdx.y为0和1, 此时为第一个warp, 那么对于(0,0)和(1,15), 他们都是bank 0, 产生冲突. 所以为了解决这个问题, 要把shared memory的大小调整为$32\*32$, 这样通过添加padding就能避免bank conflict发生了.
```cuda
#define TILE_DIM 32
__global__ void transpose_shared_optimized(const float* matrix_input, float* matrix_output, const unsigned height, const unsigned width) {
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

__global__ void transpose_shared_optimized_x4(float* matrix_input, float* matrix_output, const unsigned height, const unsigned width) {
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
```

第一份代码的block维度为(32, 8), 一个warp需要处理TILE中四行的数据; 第二份代码则是用了向量化加速, block维度为(8,32), 每8个thread读取一行数据(一次读四个). 但是这两份代码的性能几乎没有差异, 说明向量化对于共享内存来说不是一个很关键的优化手段. 而且说实话, 向量化应该是一种作弊的优化手段, 因为它要求参数中的指针不能是const的, 但是cublas的指针全部是const类型.

此时性能已经达到cublas版本的91%, 只比它慢1ms了. 还剩下最后一个优化方法, async

## shared async

```cuda
__global__ void transpose_shared_optimized_async(const float* matrix_input, float* matrix_output, const unsigned height, const unsigned width) {
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
```

使用了memcpy_async. 这种情况下, memcpy_async只能省去数据在寄存器的过程, 这部分过程时间本身就很短. 

和buclas的矩阵转置相比, 目前有两个指标有区别, 一个是Long Scoreboard Stalls, 另一个是Barrier Stalls. 具体一点, 二者相加有每个warp有50 cycles的时间在等待, cublas则只有20. 还有另外一个区别, cublas的实现所用的warp数是我的实现的四分之一, 在block大小一致的情况下, 说明他一个block处理的tile大小是64*64, 下个版本朝着这个方向优化.

## shared x64

一个TILE的大小为64*64, 但是block维度仍然保持(32,8), grid的两个维度全部减半, 在warp数上和cublas版本保持一致

```cuda
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
```

这个版本的代码实现了cublas版本99.5%的性能, 可以说几乎一样了. 

为了更好的对比, 我又写了一版不使用shared memory, tile大小为$64\*64$的naive版本, 性能远远落后该版本, 证明了shared memory在这种一个thread处理多个数据的情况会比较有用

## 总结

从transpose这一个kernel的优化历程, 可以总结出一系列对于transpose这种计算压力很小但是内存压力比较大的kernel的优化方法:
1. 处理矩阵, 使用二维TILE, 否则cache抖动会产生大量的冗余读取/写入, 严重影响性能
2. 处理非合并的写入优先于处理非合并的读取
3. Shared Memory是一个让读取和写入都能合并的好解决方案
4. 在transpose这个kernel中, 让每个thread处理16个单精度元素能获得比较高的性能
5. 注意bank conflict, 添加padding可以有效解决
6. memcpy_async在这种不怎么需要计算的kernel中不太有用
7. 向量化读取/写入有奇效, 但是也算是一种cheating
