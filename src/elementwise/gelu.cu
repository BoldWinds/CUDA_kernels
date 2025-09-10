#include "timer.h"

#define SQRT_2_DIV_PI_F32 0.7978845608028653f
#define SQRT_2_DIV_PI_F16 __float2half(0.7978845608028653f)

#define HALF_POINT_5 __float2half(0.5f)
#define HALF_1 __float2half(1.0f)
#define HALF_POINT_44715 __float2half(0.44715f)

__device__ float gelu_single(const float x){
    return 0.5f * x * (1.0f  + atanf(SQRT_2_DIV_PI_F32*(x + 0.44715f * powf(x,3))));
}

__device__ float gelu_half(const half x){
    return HALF_POINT_5 * x * (HALF_1  + htanh(SQRT_2_DIV_PI_F16*(x + HALF_POINT_44715 * x * x * x)));
}


__global__ void gelu_naive(const float* x, float* y, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = gelu_single(x[idx]);
    }
}

__global__ void gelu_four(float* x, float* y, unsigned n) {
    unsigned idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        float4 reg_x = reinterpret_cast<float4 *>(&x[idx])[0];
        float4 reg_y;
        reg_y.x = gelu_single(reg_x.x);
        reg_y.y = gelu_single(reg_x.y);
        reg_y.z = gelu_single(reg_x.z);
        reg_y.w = gelu_single(reg_x.w);
        reinterpret_cast<float4 *>(&y[idx])[0] = reg_y;
    }
}

__global__ void gelu_half_naive(const half *x, half *y, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = gelu_half(x[idx]);
    }
}

__global__ void gelu_half_two(half *x, half *y, unsigned n) {
    unsigned idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        half2 reg_x = reinterpret_cast<half2*>(&x[idx])[0];
        half2 reg_y(gelu_half(reg_x.x), gelu_half(reg_x.y));
        reinterpret_cast<half2*>(&y[idx])[0] = reg_y;
    }
}


int main(){
    Timer* timer = new Timer();
    unsigned threadsPerBlock = 256;
    unsigned blocksPerGrid;
    double duration = 0.0;
    const unsigned n = 1<<30;

    float *x, *y;
    CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));

    // ---------------------------------
    // naive
    // ---------------------------------
    {
        blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        for(int i = 0; i < 10; i++) {
            generateRandomData(x, n);
            timer->start();
            gelu_naive<<<blocksPerGrid, threadsPerBlock>>>(x, y, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
        }
        std::cout << "naive: " << duration/10 << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // four
    // ---------------------------------    
    {
        blocksPerGrid = (n/4 + threadsPerBlock - 1) / threadsPerBlock;
        for(int i = 0; i < 10; i++) {
            generateRandomData(x, n);
            timer->start();
            gelu_four<<<blocksPerGrid, threadsPerBlock>>>(x, y, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
        }
        std::cout << "four: " << duration/10 << std::endl;
        duration = 0.0;
    }


    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));

    half *half_x, *half_y;
    CUDA_CHECK(cudaMalloc(&half_x, n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&half_y, n * sizeof(half)));

    // ---------------------------------
    // half naive
    // ---------------------------------
    {
        blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        for(int i = 0; i < 10; i++) {
            generateRandomData(half_x, n);
            timer->start();
            gelu_half_naive<<<blocksPerGrid, threadsPerBlock>>>(half_x, half_y, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
        }
        std::cout << "naive half: " << duration/10 << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // half Two
    // ---------------------------------
    {
        blocksPerGrid = (n/2 + threadsPerBlock - 1) / threadsPerBlock;
        for(int i = 0; i < 10; i++) {
            generateRandomData(half_x, n);
            timer->start();
            gelu_half_two<<<blocksPerGrid, threadsPerBlock>>>(half_x, half_y, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
        }
        std::cout << "two half: " << duration/10 << std::endl;
        duration = 0.0;
    }


    CUDA_CHECK(cudaFree(half_x));
    CUDA_CHECK(cudaFree(half_y));

    return 0;
}