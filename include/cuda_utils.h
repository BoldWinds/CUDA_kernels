#pragma once
#include <iostream>
#include <cuda_runtime.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }