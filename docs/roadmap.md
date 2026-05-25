# Project Roadmap

This project is a CUDA kernel optimization portfolio. The priority is to keep adding representative deep learning operators, document the optimization process, and make the results easy to discuss in resumes and interviews.

## Near Term

- Improve benchmark output formatting so each executable prints a clear baseline, optimized result, speedup, and correctness status.
- Add small correctness checks for edge cases such as non-multiple-of-4 input sizes and non-square matrices.
- Complete the softmax optimization notes with a clearer explanation of numerical stability and reduction strategy.
- Add a concise performance table to each operator note.

## Kernel Backlog

- LayerNorm
- RMSNorm
- bias + activation fusion
- row-wise softmax
- tiled attention prototype
- SGEMM with Tensor Cores / WMMA
- reduction variants for FP16 and mixed precision

## Benchmark Backlog

- Record GPU model, CUDA version, clock setting, input shape, and baseline for each result.
- Add Nsight Compute screenshots or exported metric summaries for selected kernels.
- Compare selected kernels with cuBLAS, cuDNN, CUTLASS, or PyTorch where appropriate.
- Keep benchmark claims conservative and reproducible.

## Presentation Backlog

- Add a top-level performance summary table after more results are standardized.
- Add diagrams for SGEMM tiling, transpose memory access, and reduction hierarchy.
- Keep Chinese deep-dive notes for detailed reasoning, while keeping the main README concise and English-facing.
