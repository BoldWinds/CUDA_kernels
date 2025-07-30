#include "kernels.h"

__global__ void sgemm_naive(const float* A, const float* B, float* C, int m, int n, int k, int lda, int ldb, int ldc) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if(row < m && col < n){
        double temp = 0;
        for(int i = 0; i < k; i++){
            // A[row, i] * B[i, col]
            temp += A[row * lda + i] * B[col + i * ldb];
        }
        C[row * ldc + col] = temp;
    }
}

inline void launch_sgemm_naive(cudaStream_t stream, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc){
    dim3 blockSize = dim3(16,16);
    dim3 gridSize = dim3((n + blockSize.x - 1)/blockSize.x, (m + blockSize.y - 1)/ blockSize.y);
    sgemm_naive<<<gridSize, blockSize, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc);
}