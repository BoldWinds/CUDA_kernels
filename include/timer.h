#pragma once

#include "cuda_utils.cuh"
#include <chrono>

class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

    double clear() {
        auto duration = elapsed();
        start_ = end_;
        return duration;
    }

private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }
    void start() {
        CUDA_CHECK(cudaEventRecord(start_, 0));
    }
    float stop() {
        float elapsed_ms;
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }
private:
    cudaEvent_t start_, stop_;
};