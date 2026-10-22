# 02 Performance regimes

A GPU step is limited by one of three things (Horace He's framing):

| Regime | Time goes to | Typical operations | What helps |
|---|---|---|---|
| **Compute-bound** | arithmetic (FLOPs) | large matrix multiplications | fp16 Tensor Cores, a faster GPU |
| **Memory-bandwidth-bound** | moving data between the GPU's VRAM and its compute units (**inside** the GPU, not CPU → GPU) | element-wise ops: add, GELU, LayerNorm, dropout, softmax; the optimizer update | fusing several ops into one kernel, so data is read and written once |
| **Overhead-bound** | the CPU-side cost of launching kernels | many small kernels on small inputs | more work per kernel (a larger batch), fewer kernels (fusion, CUDA Graphs) |

**Memory-bound example** (RTX 3050): `x + 1.0` on a 4096×4096 tensor reads and writes 134 MB in 0.73 ms, ~180 GB/s, close to the card's bandwidth. The single addition per number costs nothing; moving the numbers is the time.

## How to tell which regime you are in
Increase the work and watch the time:
- time grows **proportionally** → compute- or memory-bound;
- time barely grows → **overhead-bound** (the fixed cost dominates).

## Measured on DeBERTa-v3-base (RTX 3050, ~24-token inputs, forward + backward)

| batch | fp32 ms | fp16 ms | fp16 speed-up |
|---|---|---|---|
| 8 | 79.7 | 78.2 | 1.02× |
| 16 | 136.3 | 74.8 | 1.82× |
| 32 | 247.7 | 104.8 | 2.36× |
| 64 | out of memory | 181.8 | — |

- **fp32 is mostly compute-bound:** batch 16 → 32 (2× work) costs 1.8× the time.
- **fp16 makes the compute ~2–3× faster, which exposes the fixed overhead:** batch 8 → 16 costs nothing extra (78 → 75 ms). That is the **overhead floor**, ~75 ms per forward + backward.
- **Lesson:** an optimization that shrinks compute only helps until you hit the overhead floor. Below it, a smaller batch per GPU does not get faster.

## Time per epoch
`time per epoch = (segments / batch size) × time per step`. Whenever the step grows **less** than the batch, larger batches make epochs shorter. With fp16 above, batch 16 → 64 is 4× the work for 2.4× the time, so ~40% less time per epoch. A larger batch also changes training (fewer, larger updates; the LR may need re-tuning), so it must be validated like any hyperparameter.

**Sources:** Horace He, *Making Deep Learning Go Brrrr From First Principles* (https://horace.io/brrr_intro.html); NVIDIA, *GPU Performance Background* (https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html); PyTorch blog, *Accelerating PyTorch with CUDA Graphs* (https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).
