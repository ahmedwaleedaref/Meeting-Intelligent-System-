# 01 CPU and GPU execution

**Two processors, two memories.** Python runs on the CPU. Tensors on a GPU live in the GPU's own memory (VRAM). The GPU runs **kernels**: one small compiled program per operation (`a @ b`, `x + 1`, a LayerNorm…). Every operation on a GPU tensor means the CPU **launches** a kernel and the GPU **executes** it.

## Launching is asynchronous
The CPU puts the kernel in a queue (a CUDA stream) and continues with the next Python line at once. The GPU works through the queue in order. Normally the CPU runs ahead, preparing the next batch while the GPU computes the current one.

Measured on an RTX 3050, 50 multiplications of 4096×4096 matrices:

| | time |
|---|---|
| Python loop finished | 105 ms |
| GPU finished the work | 1,862 ms |

## Synchronisation points
An operation that needs a **value on the CPU** must wait until the GPU has finished everything queued before it: `.item()`, `.cpu()`, `.tolist()`, `.numpy()`, `print(tensor)`, `if tensor:`, `torch.cuda.synchronize()`. While the CPU waits, it cannot prepare the next work. If this happens every step, the CPU and GPU take turns instead of overlapping.

## Consequence for timing
`time.perf_counter()` around GPU code measures how long it took to **queue** the work. To measure the work itself, call `torch.cuda.synchronize()` before each timestamp, or use `torch.cuda.Event`.

## Launch cost
Every kernel launch has a fixed CPU-side cost: Python dispatch, PyTorch choosing the right kernel for the device and dtype, the CUDA driver. Measured on the RTX 3050: about 10 µs for a bare `x + 1.0`. Inside a real model each operation also passes through the module's Python code and autograd's bookkeeping. A forward + backward of DeBERTa-v3-base launches **1,742 kernels**, and its fixed cost came out at roughly 75 ms (see [02](02-performance-regimes.md)).

**Sources:** PyTorch, *CUDA semantics*, "Asynchronous execution" (https://docs.pytorch.org/docs/stable/notes/cuda.html); *Performance Tuning Guide*, "Avoid unnecessary CPU-GPU synchronization" (https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
