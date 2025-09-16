#include "timer.h"
#include <float.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void reduce_max(const float* input, const unsigned n, float* output) {
    __shared__ float smem[32];

    auto grid = cg::this_grid();
    const unsigned offset = grid.thread_rank();
    const unsigned stride = grid.num_threads();

    float max = FLT_MIN;
    for(int i = offset; i < n; i+=stride){
        max = fmaxf(max, input[i]);
    }

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    max = cg::reduce(warp, max, cg::greater<float>());

    if(warp.thread_rank() == 0) {
        smem[warp.meta_group_rank()] = max;
    }
    __syncthreads();

    if(warp.meta_group_rank() == 0) {
        max = warp.thread_rank() < warp.meta_group_size() ? smem[warp.thread_rank()] : FLT_MIN;
        max = cg::reduce(warp, max, cg::greater<float>());
        if(warp.thread_rank() == 0) output[grid.block_rank()] = max;
    }
}

__global__ void reduce_max_x4(const float* input, const unsigned n, float* output) {
    __shared__ float smem[32];

    auto grid = cg::this_grid();
    const unsigned offset = 4 * grid.thread_rank();
    const unsigned stride = 4 * grid.num_threads();

    float max = FLT_MIN;
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
        max = warp.thread_rank() < warp.meta_group_size() ? smem[warp.thread_rank()] : FLT_MIN;
        max = cg::reduce(warp, max, cg::greater<float>());
        if(warp.thread_rank() == 0) output[grid.block_rank()] = max;
    }
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
            int index;
            cublasIsamax(handle, n, data, 1, &index);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            float max;
            CUDA_CHECK(cudaMemcpy(&max, &data[index-1], sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cout << max << std::endl;
        }
        cublasDestroy(handle);
        std::cout << "cuBLAS: " << duration / max_run << " ms" << std::endl;
        duration = 0.0;
    }


    // ---------------------------------
    // max
    // ---------------------------------
    {
        const unsigned temp_size = 492*2;
        float *d_max, *d_temp, max;
        CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp, temp_size * sizeof(float)));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            reduce_max<<<temp_size, 256>>>(data, n, d_temp);
            reduce_max<<<1, 256>>>(d_temp, temp_size, d_max);
            CUDA_CHECK(cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            std::cout << max << std::endl;
        }
        std::cout << "reduce max: " << duration/max_run << " ms" << std::endl;
        duration = 0.0;
        CUDA_CHECK(cudaFree(d_max));
        CUDA_CHECK(cudaFree(d_temp));
    }

    // ---------------------------------
    // max x4
    // ---------------------------------
    {
        const unsigned temp_size = 246;
        float *d_max, *d_temp, max;
        CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_temp, temp_size * sizeof(float)));
        for(int i = 0; i < max_run; i++) {
            timer->start();
            reduce_max_x4<<<temp_size, 256>>>(data, n, d_temp);
            reduce_max_x4<<<1, 256>>>(d_temp, temp_size, d_max);
            CUDA_CHECK(cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            duration += timer->stop();
            std::cout << max << std::endl;
        }
        std::cout << "reduce max x4: " << duration/max_run << " ms" << std::endl;
        duration = 0.0;
        CUDA_CHECK(cudaFree(d_max));
        CUDA_CHECK(cudaFree(d_temp));
    }
    CUDA_CHECK(cudaFree(data));
}