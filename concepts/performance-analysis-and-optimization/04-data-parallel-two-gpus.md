# 04 Data-parallel training on two GPUs

`accelerate launch --num_processes 2` starts **two separate Python processes**, one per GPU. Each runs the whole training script: it loads its own copy of the model and builds its own optimizer and DataLoader. That's why the load report is printed twice. The two processes agree on one thing: **their gradients**.

## One training step with `split_batches=True`

```
sampler yields the batch of 32 segments [s1 … s32]  (same seed → same batch in both processes)
                 │
      ┌──────────┴──────────┐
 process 0 / GPU 0      process 1 / GPU 1
 segments s1 … s16      segments s17 … s32
 forward  (own copy)    forward  (own copy)
 backward → gradients   backward → gradients
      └──── all-reduce: average the two gradient sets ────┘   ← the GPUs exchange 184M numbers (~0.74 GB)
 clip, AdamW step       clip, AdamW step     ← identical gradients → identical update → copies stay identical
```

- **The update is the same as on one GPU with batch 32.** The average of the two halves' mean gradients is the mean gradient of all 32. That's why the LR schedule and the results stay comparable. Without `split_batches`, each GPU would take its own 32, and one step would see 64.
- **The all-reduce** (sum the gradients across GPUs, then divide) is the only communication. DDP starts sending finished gradients in **buckets** while the backward pass is still running, so part of the communication overlaps with computation.
- **At the start,** DDP copies GPU 0's weights to GPU 1, so both start identical even if their random initialisations differed.
- **Validation runs on GPU 0 only.** GPU 1 waits at `accelerator.wait_for_everyone()`, a barrier both processes must reach.
- **Logged losses are averaged over both GPUs** with `accelerator.reduce`, another collective operation, which both processes must call at the same point.

## What it costs, and why it didn't speed up this run
- **Each GPU gets only 16 segments.** In fp16 that is at the overhead floor ([02](02-performance-regimes.md): 74.8 ms at batch 16 vs 78.2 ms at batch 8 on the RTX 3050). Halving the work per GPU saves almost no time.
- **The all-reduce moves ~0.74 GB of gradients every step.** Kaggle's two T4s are connected through PCIe, not a fast GPU-to-GPU link, so this is a real cost that a single GPU doesn't have.
- **Hypothesis for the T4 (to be measured):** fp16's gain was cancelled by the overhead floor plus the gradient exchange. If so, the better use of two GPUs here is **two independent runs** (one seed per GPU, `CUDA_VISIBLE_DEVICES=0` / `1`). There's no communication then, and twice as many runs per hour.

**Sources:** Li et al. 2020, *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* (https://arxiv.org/abs/2006.15704); PyTorch, *DDP notes* (https://docs.pytorch.org/docs/stable/notes/ddp.html); Accelerate, *Accelerator* reference, `split_batches` (https://huggingface.co/docs/accelerate/package_reference/accelerator).
