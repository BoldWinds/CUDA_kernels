#include "kernels.h"

__global__ void matrix_add_naive(const float* A, const float* B, float* C, int m, int n, int lda, int ldb, int ldc) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < m && col < n){
        C[row * ldc + col] = A[row * lda + col] + B[row * ldb + col];
    }
}

inline void launch_matrix_add_naive(const float* d_A, const float* d_B, float* d_C, int m, int n, int lda, int ldb, int ldc, cudaStream_t stream) {
    dim3 blockSize = dim3(16, 16);
    dim3 gridSize = dim3((n + blockSize.x -1)/ blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    matrix_add_naive<<<gridSize, blockSize, 0, stream>>>(d_A, d_B, d_C, m, n, lda, ldb, ldc);
}

