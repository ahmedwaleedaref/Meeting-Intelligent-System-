# Performance analysis and optimization

How a training step spends its time on a GPU, and what each optimization in `src/smi/models/model.py` changes. Used by [the phase-C report](../../tasks/C-baselines/REPORT.md), §6.

| Note | Question it answers |
|---|---|
| [01 CPU and GPU execution](01-cpu-gpu-execution.md) | Who does what, why GPU work is asynchronous, what a synchronisation point is |
| [02 Performance regimes](02-performance-regimes.md) | Is a step limited by compute, memory bandwidth or overhead, and how to tell |
| [03 The training loop before and after](03-training-loop-before-after.md) | What each of the five optimizations changed in the code |
| [04 Data-parallel training on two GPUs](04-data-parallel-two-gpus.md) | What the two GPUs actually do in every step, and what it costs |
| [05 Measured on the T4, one change at a time](05-measured-on-t4.md) | The results: fp32 → fp16 → fused AdamW, per phase, and which hypotheses they corrected |
| [06 How to measure a training step](06-how-to-measure.md) | The method: one change at a time, predictions first, syncs, warm-up, side effects |
| [From zero to here](from-zero-to-here.md) | A self-study path through all of the above: reading per level, hands-on exercises, the tools |

**Reading order for the sources:**
1. Horace He, *Making Deep Learning Go Brrrr From First Principles*: https://horace.io/brrr_intro.html
2. PyTorch, *CUDA semantics*, section "Asynchronous execution" only: https://docs.pytorch.org/docs/stable/notes/cuda.html
3. PyTorch, *Performance Tuning Guide*: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

**Method rule learned the hard way:** change one thing at a time and measure each change. Five changes measured only together gave "no speed-up" and no way to tell which change helped and which cancelled it.
