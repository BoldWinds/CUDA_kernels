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
__global__ void transpose_naive(float* matrix_input, float* matrix_output, unsigned rows, unsigned cols){
    unsigned row_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned col_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if(col_idx < cols && row_idx < rows) {
        matrix_output[col_idx * rows + row_idx] = matrix_input[row_idx * cols + col_idx];
    }
}
```

kernel用时15.03ms，计算利用率15.6%，内存利用率62.7%，全面低于CUBLAS。观察Grid Size，发现使用的线程数是cublas的16倍，所以应该尝试一个thread处理更多的数据，提高内存利用率

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



