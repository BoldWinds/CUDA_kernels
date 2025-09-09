#include "timer.h"

__global__ void sigmoid_naive(float* x, float* y, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) y[idx] = 1 / (1 +  expf(-1 * x[idx]));
}

__global__ void sigmoid_arithmetic(float* x, float* y, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) y[idx] = __fdividef(1, 1 + __expf(-1 * x[idx]));
}

int main(){
    float *x, *y;
    const unsigned n = 1<<30;
    CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
    Timer* timer = new Timer();
    unsigned threadsPerBlock = 256;
    unsigned blocksPerGrid;
    
    // ---------------------------------
    // naive
    // ---------------------------------
    
    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    double duration_naive = 0.0;
    for(int i = 0; i < 10; i++) {
        generateRandomData(x, n);
        timer->start();
        sigmoid_naive<<<blocksPerGrid, threadsPerBlock>>>(x, y, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        timer->stop();
        duration_naive += timer->elapsed();
    }
    std::cout << "naive: " << duration_naive/10 << std::endl;

    // ---------------------------------
    // arithmetic opt
    // ---------------------------------
    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    double duration_arithmetic = 0.0;
    for(int i = 0; i < 10; i++) {
        generateRandomData(x, n);
        timer->start();
        sigmoid_arithmetic<<<blocksPerGrid, threadsPerBlock>>>(x, y, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        timer->stop();
        duration_arithmetic += timer->elapsed();
    }
    std::cout << "arithmetic: " << duration_arithmetic/10 << std::endl;

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));

    return 0;
}