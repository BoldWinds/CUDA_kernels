#pragma once

#include "cuda_utils.h"

class CudaTimer {
public:
    CudaTimer() {
        CHECK_CUDA_ERROR(cudaEventCreate(&start_));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        CHECK_CUDA_ERROR(cudaEventDestroy(start_));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop_));
    }
    void start() {
        CHECK_CUDA_ERROR(cudaEventRecord(start_, 0));
    }
    float stop() {
        float elapsed_ms;
        CHECK_CUDA_ERROR(cudaEventRecord(stop_, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }
private:
    cudaEvent_t start_, stop_;
};