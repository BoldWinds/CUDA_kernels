#include "timer.h"

// one thread one element
__global__ void elementwise_add_naive(const float* a, const float* b, float* c, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// one thread four elements
__global__ void elementwise_add_four(const float* a, const float* b, float* c, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total_threads = blockDim.x * gridDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        c[idx + total_threads] = a[idx + total_threads] + b[idx + total_threads];
        c[idx + 2 * total_threads] = a[idx + 2 * total_threads] + b[idx + 2 * total_threads];
        c[idx + 3 * total_threads] = a[idx + 3 * total_threads] + b[idx + 3 * total_threads];
    }
}

// one thread four elements but vectorize
__global__ void elementwise_add_vectorize(float* a, float* b, float* c, unsigned n) {
    unsigned idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        float4 reg_a = (reinterpret_cast<float4 *>(a + idx))[0];
        float4 reg_b = (reinterpret_cast<float4 *>(b + idx))[0];
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        (reinterpret_cast<float4 *>(c + idx))[0] = reg_c;
    }
}

int main(){
    float *a, *b, *c;
    const unsigned n = 1<<30;
    CUDA_CHECK(cudaMalloc(&a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, n * sizeof(float)));
    Timer* timer = new Timer();
    
    // ---------------------------------
    // naive
    // ---------------------------------
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    double duration_naive = 0.0;
    for(int i = 0; i < 10; i++) {
        generateRandomData(a, n);
        generateRandomData(b, n);
        timer->start();
        elementwise_add_naive<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer->stop();
        duration_naive += timer->elapsed();
    }
    std::cout << "naive: " << duration_naive/10 << std::endl;

    // ---------------------------------
    // 一个线程处理多个数据
    // ---------------------------------
    threadsPerBlock = 256;
    blocksPerGrid = (n + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
    double duration_four = 0.0;
    for(int i = 0; i < 10; i++) {
        generateRandomData(a, n);
        generateRandomData(b, n);
        timer->start();
        elementwise_add_four<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer->stop();
        duration_four += timer->elapsed();
    }
    std::cout << "four: " << duration_naive/10 << std::endl;

    // ---------------------------------
    // 一个线程处理多个数据(向量化)
    // ---------------------------------
    threadsPerBlock = 256;
    blocksPerGrid = (n + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
    double duration_vectorize = 0.0;
    for(int i = 0; i < 10; i++) {
        generateRandomData(a, n);
        generateRandomData(b, n);
        timer->start();
        elementwise_add_vectorize<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer->stop();
        duration_vectorize += timer->elapsed();
    }
    std::cout << "vectorize: " << duration_naive/10 << std::endl;

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));

    return 0;
}