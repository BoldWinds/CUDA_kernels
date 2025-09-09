# elementwise_add

## elementwise_add_naive

```cuda
__global__ void elementwise_add_naive(const float* a, const float* b, float* c, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

最简单的按元素加法kernel，对于$2^{30}$的单精度数据，15ms计算完毕。计算吞吐率15%，内存吞吐率94%，每个thread使用16个寄存器，在理论100%的利用率下达成了78%的利用率。
怎么优化？
1. 这个kernel是内存受限的
2. 对于3090，一个thread使用40个以下的寄存器是可以打到理论100的占用率的，所以要一个thread计算更多的数据

## elementwise_add_four

```cuda
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
```

一个thread处理四个元素。这样做提升了10%左右的计算利用率，但是性能几乎没有任何提升，为什么？
因为这个kernel的瓶颈在于把数据从全局内存中读取然后再写入全局内存的过程，而并不是计算的过程。继续优化就应当注重别的方面：这个按元素加法怎么配合别的kernel？能不能使用半精度？因为半精度相当于读写同样大小的全局内存但是处理了两倍的数据。
如果实在想对这个kernel进行提升，还可以向量化

## elementwise_add_vectorize

```cuda
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
```

然而是从结果是，没有实质的证据能证明向量化能带来什么性能提升


## 总结

以上的几个写法，在最终性能上几乎没有差别，对于这种简单的计算，就不要费心思优化kernel本身了

# sigmoid

$$
F(x) = \frac{1}{1+e^{-x}}
$$

仍然是elementwise类型的计算，写了两个版本的代码：
```cuda
__global__ void sigmoid_naive(float* x, float* y, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) y[idx] = 1 / (1 + exp(-1 * x[idx]));
}

__global__ void sigmoid_arithmetic(float* x, float* y, unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) y[idx] = __fdividef(1, 1 + __expf(-1 * x[idx]));
}
```

对于$2^{30}$个浮点数，前者用时约10.09ms，后者用时约10.07ms，几乎起不到加速，因为elementwise类型的kernel主要瓶颈还是在访存上，前面讨论过了。
