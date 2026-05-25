# CUDA Kernel Optimization Lab

> A hands-on CUDA kernel optimization project for deep learning operators.

This repository collects CUDA kernels I implemented, benchmarked, and profiled while studying GPU performance engineering. The goal is not to build a production CUDA library. It is a growing performance portfolio that shows how common deep learning operators can be optimized step by step, from naive implementations to more hardware-aware kernels.

The codebase currently focuses on memory access patterns, shared memory tiling, warp-level reduction, vectorized load/store, register tiling, and profiling-driven optimization with Nsight Compute.

## Highlights

- Implemented common CUDA operators used in deep learning workloads:
  - elementwise add, sigmoid, GELU
  - reduce sum, reduce max
  - softmax
  - matrix transpose
  - SGEMM
- Compared custom kernels against CUDA library baselines such as cuBLAS where applicable.
- Used Nsight Compute metrics to analyze bottlenecks such as memory throughput, bank conflict, MIO throttle, long scoreboard stall, tail effect, and uncoalesced global access.
- Documented the optimization reasoning in Chinese deep-dive notes for each operator family.
- Kept the project lightweight so new kernels and experiments can be added quickly.

## Kernel Gallery

| Area | Kernels | Optimization focus | Notes |
| --- | --- | --- | --- |
| Elementwise | add, sigmoid, GELU | memory bandwidth, vectorized access, `half2` | GELU includes `float4` and `half2` experiments. |
| Reduction | sum, max | grid-stride loop, warp shuffle, cooperative groups, vectorized load | Reduce sum/max are compared with cuBLAS-style baselines. |
| Softmax | softmax | reduction + numerical stability | Work in progress for more optimized fused implementations. |
| Transpose | naive, x4, shared-memory tiled, async variant | coalesced access, shared memory padding, bank conflict reduction | Several versions explore the tradeoff between global memory and shared memory behavior. |
| SGEMM | naive tiled, shared-memory tiled, thread-tile, 64x64 tile | shared memory reuse, register tiling, compute intensity | Incremental optimization path from basic tiling to larger block tiles. |

## Performance Snapshot

These are local experimental results from the current notes. They are intended to show the optimization direction rather than serve as final benchmark claims.

| Kernel | Current observation |
| --- | --- |
| GELU FP32 | `float4` version improved memory throughput by about 12.8%, reducing runtime from about 11.41 ms to 10.11 ms on the tested workload. |
| GELU FP16 | `half2` version improved memory throughput by about 25%, reducing runtime from about 6.90 ms to 5.08 ms. |
| Reduce sum | After fixing tail effect and uncoalesced access, the optimized version is close to the cuBLAS reference on the tested GPU. |
| Reduce max | The vectorized `x4` implementation reaches similar performance to the cuBLAS-style baseline in the current experiment. |
| Transpose | The optimized transpose experiments reach around 90%+ of the cuBLAS reference in the documented setup; the simple `x4` version performs especially well because it preserves high memory throughput. |
| SGEMM | The implementation evolves from a basic shared-memory version to register/thread-tile variants, reaching about 20%+ of cuBLAS performance in the documented experiments. |

Hardware and workload assumptions are recorded in the operator notes where available. The project currently targets CUDA architecture `86` in CMake, matching the author's local RTX 30-series development environment.

## Technical Deep Dives

The detailed optimization notes are written in Chinese and kept next to the kernels:

- [Elementwise notes](src/elementwise/elementwise.md)
- [Reduction notes](src/reduce/reduce.md)
- [Transpose notes](src/transpose/transpose.md)
- [GEMM notes](src/gemm/gemm.md)
- [Project roadmap](docs/roadmap.md)

## Repository Layout

```text
.
├── include/
│   ├── cuda_utils.cuh
│   └── timer.h
├── src/
│   ├── elementwise/
│   ├── reduce/
│   ├── transpose/
│   └── gemm/
├── docs/
│   └── roadmap.md
└── CMakeLists.txt
```

Each operator directory contains CUDA source files and a Markdown note explaining the implementation and profiling observations.

## Build

Requirements:

- CUDA Toolkit
- CMake 3.24+
- A CUDA-capable NVIDIA GPU

Build with CMake:

```bash
cmake -S . -B build
cmake --build build -j
```

Run an executable from the build output directory. The exact target names are defined in the operator-level `CMakeLists.txt` files.

```bash
./build/bin/gelu
./build/bin/reduce_sum
./build/bin/transpose
./build/bin/sgemm
```

## Optimization Topics Covered

- Global memory coalescing
- Grid-stride loops
- Vectorized memory access with `float4`
- Half precision vectorization with `half2`
- Shared memory tiling and padding
- Shared memory bank conflict analysis
- Warp-level reduction with shuffle and cooperative groups
- Register tiling for SGEMM
- Profiling-driven iteration with Nsight Compute

## Roadmap

This project is designed to keep growing as a CUDA operator portfolio. Planned directions include:

- Tensor Core / WMMA GEMM
- LayerNorm and RMSNorm
- more complete Softmax optimization
- fused elementwise + reduction kernels
- FlashAttention-style tiled attention kernels
- CUTLASS comparison baselines
- cleaner benchmark result tables and profiling screenshots

See [docs/roadmap.md](docs/roadmap.md) for more detail.

## Project Philosophy

This is a learning-first, profiling-first CUDA project. The most important output is not only the final kernel, but also the optimization path: what bottleneck was found, what kernel change was tried, what metric changed, and what tradeoff remained.

That makes the repository suitable as a resume project and interview discussion material: each operator is a concrete entry point for talking about GPU architecture, memory hierarchy, CUDA programming, and performance analysis.
