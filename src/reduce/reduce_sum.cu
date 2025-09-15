#include "timer.h"
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void data_process(float* data, const unsigned n) {
    const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n) data[tid] = fabs(data[tid]);
}

__global__ void reduce_sum_naive(const float* data, const unsigned n, float* output){
    __shared__ float smem[256];
    const unsigned elementsPerBlock = CEIL(n, gridDim.x);
    const unsigned offset = blockIdx.x * elementsPerBlock;

    // sum
    float sum = 0.0;
    #pragma unroll
    for(unsigned i = threadIdx.x; i < elementsPerBlock; i+=blockDim.x){
        unsigned const global_index = offset + i;
        if(global_index < n) sum += data[global_index];
    }
    smem[threadIdx.x] = (float)sum;
    __syncthreads();

    // reduce
    #pragma unroll
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    
    if(threadIdx.x == 0) output[blockIdx.x] = smem[0];
}

__global__ void reduce_sum_opt(const float* data, const unsigned n, float* output){
    __shared__ float smem[256];
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gridSize = gridDim.x * blockDim.x;
    // sum
    float sum = 0.0;
    #pragma unroll
    for(unsigned i = tid; i < n; i+=gridSize){
        sum += data[i];
    }
    smem[threadIdx.x] = sum;
    __syncthreads();

    // reduce
    #pragma unroll
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    
    if(threadIdx.x == 0) output[blockIdx.x] = smem[0];
}

__global__ void reduce_sum_shuffle_cg(const float* data, const unsigned n, float* output){
    __shared__ float smem[8];

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gridSize = gridDim.x * blockDim.x;
    // sum
    float sum = 0.0;
    #pragma unroll
    for(unsigned i = tid; i < n; i+=gridSize){
        sum += data[i];
    }
    
    // reduce
    sum = cg::reduce(warp, sum, cg::plus<float>());

    if(threadIdx.x % 32 == 0)   smem[threadIdx.x / 32] = sum;
    __syncthreads();

    if(threadIdx.x < 8) {
        sum = smem[threadIdx.x];
        auto group = cg::coalesced_threads();
        sum = cg::reduce(group, sum, cg::plus<float>());
        if(threadIdx.x == 0) output[blockIdx.x] = sum;
    }
}

__global__ void reduce_sum_shuffle(const float* data, const unsigned n, float* output){
    __shared__ float smem[8];

    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned gridSize = gridDim.x * blockDim.x;
    // sum
    float sum = 0.0;
    #pragma unroll
    for(unsigned i = tid; i < n; i+=gridSize){
        sum += data[i];
    }
    
    // reduce
    const unsigned mask = 0xffffffff;
    sum += __shfl_down_sync(mask, sum, 16);
    sum += __shfl_down_sync(mask, sum, 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);

    if(threadIdx.x % 32 == 0)   smem[threadIdx.x / 32] = sum;
    __syncthreads();

    if(threadIdx.x < 8) {
        sum = smem[threadIdx.x];
    }
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);
    if(threadIdx.x == 0) output[blockIdx.x] = sum;
}


int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "input max run times!" << std::endl;
        return -1;
    }
    int max_run = std::atoi(argv[1]);
    float *data;
    const unsigned n = 1 << 30;
    CUDA_CHECK(cudaMalloc(&data, n * sizeof(float)));
    generateRandomData(data, n);
    data_process<<<CEIL(n, 256),256>>>(data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CudaTimer* timer = new CudaTimer();
    double duration = 0.0;
    
    // ---------------------------------
    // CUBLAS
    // ---------------------------------
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        for(int i = 0; i < max_run; i++) {
            timer->start();
            float sum;
            cublasSasum(handle, n, data, 1, &sum);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            std::cout << sum << std::endl;
        }
        cublasDestroy(handle);
        std::cout << "cuBLAS: " << duration / max_run << " ms" << std::endl;
        duration = 0.0;
    }

    // ---------------------------------
    // naive
    // ---------------------------------
    {
        const unsigned temp_size = 1024;
        float *d_sum, *d_temp, sum;
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp, temp_size * sizeof(float)));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            reduce_sum_naive<<<temp_size, 256>>>(data, n, d_temp);
            reduce_sum_naive<<<1, 256>>>(d_temp, temp_size, d_sum);
            CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            std::cout << sum << std::endl;
        }
        std::cout << "naive: " << duration/max_run << " ms" << std::endl;
        duration = 0.0;
        CUDA_CHECK(cudaFree(d_sum));
        CUDA_CHECK(cudaFree(d_temp));
    }

    // ---------------------------------
    // opt
    // ---------------------------------
    {
        const unsigned temp_size = 492 * 2;
        float *d_sum, *d_temp, sum;
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp, temp_size * sizeof(float)));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            reduce_sum_opt<<<temp_size, 256>>>(data, n, d_temp);
            reduce_sum_opt<<<1, 256>>>(d_temp, temp_size, d_sum);
            CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            std::cout << sum << std::endl;
        }
        std::cout << "opt: " << duration/max_run << " ms" << std::endl;
        duration = 0.0;
        CUDA_CHECK(cudaFree(d_sum));
        CUDA_CHECK(cudaFree(d_temp));
    }

    // ---------------------------------
    // shuffle
    // ---------------------------------
    {
        const unsigned temp_size = 492 * 2;
        float *d_sum, *d_temp, sum;
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp, temp_size * sizeof(float)));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            reduce_sum_shuffle<<<temp_size, 256>>>(data, n, d_temp);
            reduce_sum_shuffle<<<1, 256>>>(d_temp, temp_size, d_sum);
            CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            std::cout << sum << std::endl;
        }
        std::cout << "shuffle: " << duration/max_run << " ms" << std::endl;
        duration = 0.0;
        CUDA_CHECK(cudaFree(d_sum));
        CUDA_CHECK(cudaFree(d_temp));
    }

    CUDA_CHECK(cudaFree(data));
}