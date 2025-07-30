#pragma once

#include "timer.h"

struct GemmArgs {
    int m, n, k;
    int lda, ldb, ldc;
};

float launch_sgemm(int version, const GemmArgs& args, bool validate = false);
void launch_sgemm_naive(cudaStream_t stream, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);

struct MatrixAddArgs {
    int m, n;
    int lda, ldb, ldc;
};

float launch_matrix_add(int version, const MatrixAddArgs& args, bool validate = false);
void launch_matrix_add_naive(const float* d_A, const float* d_B, float* d_C, int m, int n, int lda, int ldb, int ldc, cudaStream_t stream);