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

