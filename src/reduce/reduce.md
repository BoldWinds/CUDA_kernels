# Reduce

reduce 是一种聚合操作，通常用于将一个多元素的数据结构（如数组或张量）通过某种规则归约为一个更小的数据结构（通常是单个值或更小的数组）。
它广泛应用于数据处理、并行计算以及深度学习中。例如对数组进行求和 (sum)，求均值 (mean)，求最大值 (max)，还有求 softmax。
其中，**sum** 和 **softmax** 的考察频率最高。

# Reduce sum

一般来说，为了高效归约，都会选择通过两个kernel完成。那么每个thread要处理多少数据呢？
观察cuBLAS的实现，对于不同大小的规约求和，第一个kernel都由**492**个block构成，第二个kernel都只有一个block。那么优化的重点应当在于第一个kernel，第二个kernel的用时可以忽略不计。
第一个kernel选择了492个固定block比较令人奇怪，但是经过分析，3090有82个SM，每个SM分配6个block，就是492个block；而一般来说，6 blocks/SM 是一个可以保证SM繁忙，隐藏延迟的一个甜点；
而且在归约问题中，最主要的是访存的优化，其次才是计算；再多加block也会让

[reduce](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) 这篇文章提到了不少优化的方法：
- 最普通的做法，在shared memory中不断求和；优化的话就是注意避免bank conflict
- 循环展开
- 一个thread处理多个元素
- warp shuffle

## naive

```CUDA
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
```

该版本是比较简单的利用shared memory进行归约，1024个block，每个block 256个线程；计算分为两部分：
1. 合并读取，每个线程计算完该线程所负责数据的和，存入shared memory
2. 跨步归约，从blockDim.x/2为步进，每次步进除以2，直到整个block的和归约到一个数中
3. 写入结果

性能受损失最大的原因是**Tail Effect**，这是由于1024个block的数量与GPU的计算能力不匹配导致的：剩下的几十个block所花费的时间和几百个block相同。为了解决这个问题，得把block数量调整为492或者492的倍数，最终设置为了984.

解决了**Tail Effect**之后，又出现了**Uncoalesced Global Access**，这很令人奇怪，因为我们已经让临近的thread去访问临近的内存了，为什么还会出现这个问题? 这是因为每个block开始读取内存的地方没有对齐！比如现在是984个block，每个block所需要计算的元素数几乎不可能是128的整数倍，那么两个块的边界部分就会有额外的内存事务，造成非合并访问。
要想解决这个问题，就要放弃原本的访问模式。我们之前访问的方式是，把一整块数据，先分给每个block，然后每个block再处理自己的数据；现在改为，不做划分，每个线程处理自己真实线程id所对应的元素，每次步进一整个grid size的大小。这样所有的warp都是处理连续的内存空间且读取的起始地址一定是128字节对齐的，代价仅仅是尾部的部分线程少循环一次。

## opt

解决了**Tail Effect**和其带来的**Uncoalesced Global Access**后，结果如下：
```cuda
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
```

这个版本的代码已经可以做到几乎与cublas一样快了，慢大概百分之零点几

## shuffle

使用warp shuffle来替代多阶段循环的归约:
```cuda
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
```

使用cg：
```cuda
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
```

二者的性能几乎一样，并且与前面的opt版本也并无不同。主要是因为计算访存比太低了，主要优化好访存就行。

## reduce max

为了给softmax做准备，就先写了一个reduce max的kernel。
使用了reduce sum中的所有优化方法，并顺便把忘记的x4读取给补上了：
```cuda
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
```

这个版本的性能做到了与cublas一模一样

## reduce with atomic

前面的归约操作都需要启动两个kernel，但其实第二个kernel的工作量非常低下；而且两个kernel之间是通过global memory传递，有一个隐性的没参与分析的全局内存分配时间，而且全局内存传递数据也算不上多快。
为此我尝试一下把完成一次归约操作所需要调用的两次归约kernel合并为一个归约kernel，先归约然后最后的几百个数值用原子操作计算结果。

```cuda

__device__ static float atomicMax(float* address, float val) {
	int* address_as_i = (int*)address;  // address转为int指针
	int old = *address_as_i;  // address中的旧值，用int解码
	int assumed;
	do {
		assumed = old;  // assumed存储旧值
		old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__global__ void reduce_max_atomic(const float* input, const unsigned n, float* d_final_max) {
    __shared__ float smem[32];

    auto grid = cg::this_grid();
    const unsigned offset = 4 * grid.thread_rank();
    const unsigned stride = 4 * grid.num_threads();

    float max = FLT_MIN;
    for (unsigned i = offset; i < n; i += stride) {
        if(i + 3 < n){
            float4 reg = *reinterpret_cast<const float4*>(&input[i]);
            max = fmaxf(max, fmaxf(fmaxf(reg.x, reg.y), fmaxf(reg.z, reg.w)));
        } else {
            #pragma unroll
            for(; i < n; i++) {
                max = fmaxf(max, input[i]);
            }
        }
    }

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    max = cg::reduce(warp, max, cg::greater<float>());

    if(warp.thread_rank() == 0) smem[warp.meta_group_rank()] = max;
    __syncthreads();

    if(warp.meta_group_rank() == 0) {
        max = warp.thread_rank() < warp.meta_group_size() ? smem[warp.thread_rank()] : FLT_MIN;
        max = cg::reduce(warp, max, cg::greater<float>());
        if(warp.thread_rank() == 0) {
            atomicMax(d_final_max, max);
        }
    }
}
```

结果是性能几乎不受影响，甚至还更快了一点

## softmax

```cuda

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
```

没什么好说的，处理2^30数据大概需要20ms，比pytorch的300ms快太多了