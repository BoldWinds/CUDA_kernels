#include "timer.h"
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f
#define CLIP(value) fmaxf(MIN_EXP_F32, fminf(MAX_EXP_F32, value))

__constant__ float const_max, const_sum;

__global__ void softmax_max(const float* input, const unsigned n, float* output) {
    __shared__ float smem[32];

    auto grid = cg::this_grid();
    const unsigned offset = 4 * grid.thread_rank();
    const unsigned stride = 4 * grid.num_threads();

    float max = MIN_EXP_F32;
    for (unsigned i = offset; i < n; i += stride) {
        if(i + 3 < n){
            float4 reg = CONST_FLOAT4(input[i]);
            max = fmaxf(max, fmaxf(fmaxf(reg.x, reg.y), fmaxf(reg.z, reg.w)));
        }else {
            #pragma unroll
            for(; i < n; i++) {
                max = fmaxf(max, input[i]);
            }
        }
    }

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    max = cg::reduce(warp, max, cg::greater<float>());

    if(warp.thread_rank() == 0) {
        smem[warp.meta_group_rank()] = max;
    }
    __syncthreads();

    if(warp.meta_group_rank() == 0) {
        max = warp.thread_rank() < warp.meta_group_size() ? smem[warp.thread_rank()] : MIN_EXP_F32;
        max = cg::reduce(warp, max, cg::greater<float>());
        if(warp.thread_rank() == 0) output[grid.block_rank()] = max;
    }
}

__global__ void softmax_sum(const float* input, const unsigned n, float* output) {
    __shared__ float smem[32];

    auto grid = cg::this_grid();
    const unsigned offset = 4 * grid.thread_rank();
    const unsigned stride = 4 * grid.num_threads();

    float sum = 0.0f;
    for (unsigned i = offset; i < n; i += stride) {
        if(i + 3 < n){
            float4 reg = CONST_FLOAT4(input[i]);
            sum += (expf(CLIP(reg.x - const_max)) + expf(CLIP(reg.y - const_max)) + expf(CLIP(reg.z - const_max)) + expf(CLIP(reg.w - const_max)));
        }else {
            #pragma unroll
            for(; i < n; i++) {
                sum += expf(CLIP(input[i] - const_max));
            }
        }
    }

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    sum = cg::reduce(warp, sum, cg::plus<float>());

    if(warp.thread_rank() == 0) {
        smem[warp.meta_group_rank()] = sum;
    }
    __syncthreads();

    if(warp.meta_group_rank() == 0) {
        sum = warp.thread_rank() < warp.meta_group_size() ? smem[warp.thread_rank()] : 0.0f;
        sum = cg::reduce(warp, sum, cg::plus<float>());
        if(warp.thread_rank() == 0) output[grid.block_rank()] = sum;
    }
}

__global__ void softmax_normalize(const float* input, float* output, const unsigned n){
    auto grid = cg::this_grid();
    const unsigned offset = 4 * grid.thread_rank();
    const unsigned stride = 4 * grid.num_threads();

    for (unsigned i = offset; i < n; i += stride) {
        if(i + 3 < n){
            float4 reg = CONST_FLOAT4(input[i]);
            reg.x = expf(CLIP(reg.x - const_max))/const_sum;
            reg.y = expf(CLIP(reg.y - const_max))/const_sum;
            reg.z = expf(CLIP(reg.z - const_max))/const_sum;
            reg.w = expf(CLIP(reg.w - const_max))/const_sum;
            FLOAT4(output[i]) = reg;
        }else {
            #pragma unroll
            for(; i < n; i++) {
                output[i] = expf(CLIP(input[i] - const_max))/const_sum;
            }
        }
    }
}


int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "input max run times!" << std::endl;
        return -1;
    }
    int max_run = std::atoi(argv[1]);
    float *data, *output;
    const unsigned n = 1 << 30;
    CUDA_CHECK(cudaMalloc(&data, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(float)));
    generateRandomData(data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CudaTimer* timer = new CudaTimer();
    double duration = 0.0;

    {
        const unsigned temp_size = 492;
        float *d_max, *d_sum;
        
        CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            softmax_max<<<temp_size, 256>>>(data, n, output);
            softmax_max<<<1, 256>>>(output, temp_size, d_max);
            CUDA_CHECK(cudaMemcpyToSymbol(const_max, d_max, sizeof(float)));
            softmax_sum<<<temp_size, 256>>>(data, n, output);
            softmax_sum<<<1, 256>>>(output, temp_size, d_sum);
            CUDA_CHECK(cudaMemcpyToSymbol(const_sum, d_sum, sizeof(float)));
            softmax_normalize<<<temp_size, 256>>>(data, output, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
        }
        std::cout << "softmax: " << duration/max_run << " ms" << std::endl;
        duration = 0.0;
        CUDA_CHECK(cudaFree(d_max));
        CUDA_CHECK(cudaFree(d_sum));
    }

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(output));
}