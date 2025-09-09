#include "timer.h"

#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define MAX_EXP_F16 __float2half(11.089866488461016f)
#define MIN_EXP_F16 __float2half(-9.704060527839234f)

__global__ void sigmoid_naive(float* x, float* y, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    float minus_x = fminf(fmaxf(-1.0f * x[idx], MIN_EXP_F32), MAX_EXP_F32);
    if(idx < n) y[idx] = 1.0f / (1.0f +  exp(minus_x));
}

__global__ void sigmoid_arithmetic(float* x, float* y, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    float minus_x = fminf(fmaxf(-1.0f * x[idx], MIN_EXP_F32), MAX_EXP_F32);
    if(idx < n) y[idx] = __fdividef(1.0f, 1.0f + __expf(minus_x));
}

__global__ void sigmoid_half_naive(half* x, half* y, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const half one = __float2half(1.0f);
    half minus_x = __hmin(__hmax(-x[idx] , MIN_EXP_F16), MAX_EXP_F16);
    if(idx < n) y[idx] = one / (one +  hexp(minus_x));
}

__global__ void sigmoid_half_two(half* x, half* y, unsigned n) {
    unsigned idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const half one = __float2half(1.0f);
    if(idx < n) {
        half2 temp_x = reinterpret_cast<half2 *>(&x[idx])[0];
        temp_x.x = __hmin(__hmax(temp_x.x, MIN_EXP_F16), MAX_EXP_F16);
        temp_x.y = __hmin(__hmax(temp_x.y, MIN_EXP_F16), MAX_EXP_F16);
        half2 temp_y;
        temp_y.x = one / (one +  hexp(temp_x.x));
        temp_y.y = one / (one +  hexp(temp_x.y));

        reinterpret_cast<half2 *>(&y[idx])[0] = temp_y;
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
            sigmoid_naive<<<blocksPerGrid, threadsPerBlock>>>(x, y, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
        }
        std::cout << "naive: " << duration/10 << std::endl;
        duration = 0.0;
    }


    // ---------------------------------
    // arithmetic opt
    // ---------------------------------
    {
        blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        for(int i = 0; i < 10; i++) {
            generateRandomData(x, n);
            timer->start();
            sigmoid_arithmetic<<<blocksPerGrid, threadsPerBlock>>>(x, y, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
        }
        std::cout << "arithmetic: " << duration/10 << std::endl;
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
            sigmoid_half_naive<<<blocksPerGrid, threadsPerBlock>>>(half_x, half_y, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer->stop();
            duration += timer->elapsed();
        }
        std::cout << "naive half: " << duration/10 << std::endl;
        duration = 0.0;
    }


    // ---------------------------------
    // half2 opt
    // ---------------------------------
    {
        blocksPerGrid = (n/2 + threadsPerBlock - 1) / threadsPerBlock;
        for(int i = 0; i < 10; i++) {
            generateRandomData(half_x, n);
            timer->start();
            sigmoid_half_two<<<blocksPerGrid, threadsPerBlock>>>(half_x, half_y, n);
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