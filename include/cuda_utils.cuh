#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

template<typename T>
void generateRandomData(T* d_data, size_t n, unsigned long long seed = 0);

template<>
void generateRandomData<float>(float* d_data, size_t n, unsigned long long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, d_data, n);
    curandDestroyGenerator(gen);
}

template<>
void generateRandomData<double>(double* d_data, size_t n, unsigned long long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniformDouble(gen, d_data, n);
    curandDestroyGenerator(gen);
}

template<>
void generateRandomData<int>(int* d_data, size_t n, unsigned long long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerate(gen, reinterpret_cast<unsigned int*>(d_data), n);
    curandDestroyGenerator(gen);
}

__global__ void floatToHalfKernel(__half *out, const float *in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

template<>
void generateRandomData<__half>(__half* d_data, size_t n, unsigned long long seed) {
    float* d_temp_floats;
    CUDA_CHECK(cudaMalloc(&d_temp_floats, n * sizeof(float)));
    generateRandomData(d_temp_floats, n, seed);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    floatToHalfKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_temp_floats, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_temp_floats));
}