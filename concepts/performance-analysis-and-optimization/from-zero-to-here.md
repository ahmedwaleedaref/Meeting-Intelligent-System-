# From zero to here: making training loops fast

A self-study path through everything used in this folder, one level deeper than the notes, with hands-on exercises. Each level has **what to understand**, **what to read** (all links checked on 2026-10-23), **what to do by hand**, and **questions to check yourself** (answers are in notes 01–06 or in your own measurements).

The rule for every exercise: **write your prediction down before you run it.** A number you predicted tests your understanding; a number you only explain afterwards doesn't.

Most exercises fit a laptop GPU (an RTX 3050 with 4 GB works). The ones marked **[T4]** need more memory: use Kaggle.

## Setting up the exercises

The exercises are **independent of the project**: do them in a separate folder, one Python file or notebook per exercise (e.g. `ex03_queue_vs_finished.py`). Nothing in the repo is touched, and `notebooks/` stays for the project's own exploration (PROTOCOL §8).

```bash
mkdir ~/perf-exercises && cd ~/perf-exercises
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers matplotlib jupyter
```
(Or reuse the project's `.venv`, which already has `torch` and `transformers`.)

**What each exercise needs:**
- **Most of them** (levels 0–2, 4–7) need only PyTorch, random tensors and a GPU. No data, no model download.
- **The exercises with a transformer** (10, 21–24, 29) don't need trained weights: speed and memory depend only on the architecture. Build the model from a config with random weights, and feed it random token ids:
  ```python
  import torch
  from transformers import AutoConfig, AutoModelForSequenceClassification

  config = AutoConfig.for_model("bert", num_labels=7)          # BERT-base shape, no download
  model = AutoModelForSequenceClassification.from_config(config).cuda()
  input_ids = torch.randint(0, 30000, (32, 24), device="cuda")   # batch 32 , ~24 tokens like k = 0
  ```
- **Reading the source** (15, 16, 20) means PyTorch's own files in the venv you use, e.g. `.venv/lib/python3.*/site-packages/torch/optim/adam.py` and `torch/amp/grad_scaler.py`.
- **Only exercise 11 refers to the project:** rebuild the batch-size notebook from memory in your folder, then compare it with `notebooks/kaggle_batch_size_test.ipynb`.

**On a 4 GB GPU:** a full BERT- or DeBERTa-size training step with AdamW (weights + gradients + `m` + `v` ≈ 3 GB) only fits small batches in fp32. Use small batches, or shrink the model in the config, e.g. `AutoConfig.for_model("bert", num_labels=7, num_hidden_layers=4, hidden_size=512, num_attention_heads=8, intermediate_size=2048)` (28M parameters; `hidden_size` must be divisible by `num_attention_heads`). A smaller model shifts the step towards overhead-bound, and that's worth observing in its own right.

---

## Level 0: The training loop itself (prerequisite)

**Understand:** what every line of a PyTorch training step does: forward, loss, `backward()`, `optimizer.step()`, `zero_grad()`, the LR scheduler, gradient accumulation, gradient clipping.

**Read:**
- PyTorch, *Learn the Basics*, the whole series: https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- PyTorch, *Autograd mechanics*, how the graph is recorded in the forward and executed in the backward: https://docs.pytorch.org/docs/stable/notes/autograd.html

**Do:**
1. **Write a training loop from scratch** for a small MLP on random data, without copying. Then add gradient accumulation and check by hand that `grad_accum_steps=4` with batch 8 gives the same gradient as batch 32 (compare `p.grad` after one optimizer step's worth of batches).
2. **`.backward()` adds, it doesn't overwrite.** Call `loss.backward()` twice without `zero_grad()` and print a gradient. Then explain why the loop needs `zero_grad()`.

**Check yourself:** why is each batch's loss divided by its accumulation group's size? What goes wrong without it (the LR, the clipping)?

---

## Level 1: CPU and GPU: kernels, the queue, synchronisation

**Understand:**
- **A kernel** is one GPU program per operation. The CPU **launches** it, the GPU **executes** it.
- **Launches are asynchronous:** they go into a queue (a CUDA stream), and Python continues at once.
- **A synchronisation point** is anything that needs a value on the CPU, so the CPU waits for the whole queue.
- **Two timelines:** the CPU's and the GPU's. Without syncs they overlap; with a sync every step they take turns.

**Read:**
- PyTorch, *CUDA semantics*, section **"Asynchronous execution"** only: https://docs.pytorch.org/docs/stable/notes/cuda.html
- PyTorch, *Performance Tuning Guide*, "Avoid unnecessary CPU-GPU synchronization": https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Edward Yang, *PyTorch internals*: what happens between `a + b` in Python and the kernel (dispatch, autograd). Long; read it once the rest feels solid: https://blog.ezyang.com/2019/05/pytorch-internals/

**Do:**
3. **Queued vs finished.** Time 50 large matrix multiplications with `time.perf_counter()`, once without and once with `torch.cuda.synchronize()` before the second timestamp. (Note 01: 105 ms vs 1,862 ms on an RTX 3050.)
4. **Find the sync.** Write a loop with `.item()` inside, then without, and time it. Repeat with `print(tensor)`, `if tensor > 0:` and `tensor.cpu()`. Which ones are syncs?
5. **The launch floor.** Time 1,000 additions on 8×8, 256×256, 1024×1024 and 4096×4096 tensors. Find the size below which the time per kernel stops shrinking: that's the launch cost.
6. **The overlap.** Build a loop where the CPU does real work between GPU calls (e.g. tokenizing with `time.sleep(0.002)` as a stand-in). Time it with and without a sync at the end of each iteration, and explain the difference with "CPU time + GPU time" vs "the larger of the two".

**Check yourself:**
- Can backward kernels be *launched* before the forward kernels have *finished*? Can they *run* before?
- Why didn't removing `loss.item()` speed up our fp16 + fused step on the T4 (note 05)?

---

## Level 2: The memory hierarchy and the three regimes

**Understand:**
- **The path data takes:** CPU RAM →(PCIe)→ VRAM →(memory bus)→ SRAM / registers next to the compute units.
- **Arithmetic intensity:** operations per byte moved. Matrix multiplications have a lot; element-wise operations have almost none.
- **The three regimes:** compute-bound, memory-bandwidth-bound, overhead-bound. How to tell them apart: increase the work and watch the time.

**Read:**
- Horace He, *Making Deep Learning Go Brrrr From First Principles*, the core text for this level: https://horace.io/brrr_intro.html
- NVIDIA, *GPU Performance Background*: arithmetic intensity, math- vs memory-limited: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
- NVIDIA, *Matrix Multiplication Background*: why matrix sizes and multiples of 8 matter on Tensor Cores: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
- The roofline model, the picture that ties the regimes together: https://en.wikipedia.org/wiki/Roofline_model

**Do:**
7. **Measure your GPU's memory bandwidth.** Time `y = x + 1.0` on a 4096×4096 fp32 tensor, compute bytes moved ÷ time, and compare with the card's specification.
8. **Measure your GPU's compute throughput.** Time n×n matrix multiplications for n = 128 … 8192 and compute TFLOPS = 2n³ / time. Plot it. Where does it level off? Where are small matmuls far below peak, and why?
9. **Arithmetic intensity by hand.** Compute operations per byte for an addition of two vectors, a LayerNorm over 768 values, and an n×n matrix multiplication. Place each on the roofline of your GPU.
10. **The batch-size test on a small transformer.** Time forward + backward of a BERT-sized model (random weights from a config, no download) at batch 8, 16, 32 and 64, in fp32 and in fp16. Decide the regime for each. (Note 02 has the DeBERTa numbers to compare against.)

**Check yourself:**
- Why is "copying the batch to the GPU" not what Horace He means by memory-bound?
- Why can a smaller batch per GPU stop getting faster?

---

## Level 3: Measuring a training step

**Understand:** the measurement method in note 06:
- one change at a time, predictions first;
- syncs, warm-up, total vs breakdown mode;
- per token, same session;
- side effects of measurement code;
- check the default path byte for byte.

**Tools for this level, in code:**
```python
# wall time of GPU work: sync before each timestamp
torch.cuda.synchronize(); t0 = time.perf_counter()
...  # GPU work
torch.cuda.synchronize(); elapsed = time.perf_counter() - t0

# GPU-side timing with CUDA events (measures on the GPU's own clock)
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record(); ...; end.record()
torch.cuda.synchronize(); ms = start.elapsed_time(end)

# peak memory
torch.cuda.reset_peak_memory_stats()
...  # one training step
peak_gb = torch.cuda.max_memory_allocated() / 1e9
```
- `torch.cuda.Event`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html
- `torch.cuda` memory functions: https://docs.pytorch.org/docs/stable/cuda.html
- `nvidia-smi` in a terminal or notebook cell: utilisation and memory per GPU, e.g. `nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv -l 1`

**Do:**
11. **Rebuild the batch-size notebook from memory** (`notebooks/kaggle_batch_size_test.ipynb`): one step function, a warm-up, a total mode, a breakdown mode. Compare your version with the original.
12. **Reproduce the warm-up mistake.** Measure batch 32 first without a warm-up pass, then with one, and explain the difference.
13. **Reproduce the measurement side effect.** Add `scaler.get_scale()` inside the timed loop and measure the cost of the extra syncs.

**Check yourself:** why is the breakdown total higher than the total mode? What does the difference tell you?

---

## Level 4: Optimizers and fusion

**Understand:**
- **The AdamW update:** `m`, `v`, the bias correction, decoupled weight decay.
- **Where everything lives:** all of it is in VRAM.
- **Why the update is memory-bound:** ~28 bytes moved per parameter for ~10–15 operations.
- **Three implementations:**
  - **single-tensor:** a Python loop over the tensors;
  - **foreach:** one pass per line of the update, over all tensors;
  - **fused:** one pass for the whole update.
- **Why fusion saves time:** no intermediate tensors written to VRAM and read back, and fewer launches.

**Read:**
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (the AdamW paper): https://arxiv.org/abs/1711.05101
- `torch.optim.AdamW`, parameters `foreach` and `fused`: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- The code itself: `torch/optim/adam.py` in your environment, function `_multi_tensor_adam`. Count the passes, as in note 05.

**Do:**
14. **AdamW three ways.** On a 50M-parameter tensor, implement the update as a plain loop over chunks, as the separate lines of the update, and with `torch.optim.AdamW(fused=True)`. **Predict** each one's time from the bytes it moves and your measured bandwidth (exercise 7), then measure.
15. **Count passes in `_multi_tensor_adam`.** Turn your count into GB moved per step for 184M parameters, and compare with the measured 63.5 ms on the T4 (note 05).
16. **The `fused=False` trap.** Show that `AdamW(..., fused=False)` and `AdamW(...)` run different implementations: time them, and compare the updated weights after a few steps.

**Check yourself:**
- Why are the optimizer and clip costs the same at batch 32 and 64?
- Why did fused AdamW not reduce peak memory?

---

## Level 5: Mixed precision

**Understand:**
- **The fp16 format:** max ≈ 65,504, smallest normal ≈ 6·10⁻⁵, 10 bits of mantissa.
- **Why gradients underflow,** and why loss scaling fixes it.
- **fp32 master weights:** the weights stay fp32 while the computation runs in fp16.
- **What autocast runs in fp16 and what it keeps in fp32.**
- **The GradScaler loop:** scale → backward → unscale → clip → step (skipped on overflow) → update.
- **Why fused AdamW avoids the scaler's sync:** the overflow flag is a GPU tensor that the fused kernel reads itself.

**Read:**
- Micikevicius et al. 2018, *Mixed Precision Training*: https://arxiv.org/abs/1710.03740
- NVIDIA, *Train With Mixed Precision*: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
- PyTorch AMP docs, which ops autocast runs in which precision: https://docs.pytorch.org/docs/stable/amp.html
- PyTorch AMP recipe: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
- The half-precision format: https://en.wikipedia.org/wiki/Half-precision_floating-point_format

**Do:**
17. **Underflow by hand.** Take a list of small gradient-sized values (1e-3 … 1e-9), convert them to fp16, and see which become 0. Multiply by 65,536 first and repeat.
18. **Force an overflow.** Put a huge value into one gradient before `scaler.step()` and watch the step being skipped and the scale halved.
19. **fp16 speed vs shape.** Time fp16 matmuls with dimensions 1000 vs 1024. Do multiples of 8 matter on your GPU?
20. **Where the scaler syncs.** Read `torch/amp/grad_scaler.py` (`_maybe_opt_step`, and the `_step_supports_amp_scaling` branch) and find the `.item()` that fused AdamW avoids.

**Check yourself:**
- Why was fp16 *slower* on the tiny 1-layer model?
- What exactly is checked for overflow: the gradients or `v`?

---

## Level 6: Profiling

**Understand:**
- **What a profiler records:** CPU operations and GPU activities, each with start and end times.
- **GPU busy time and utilisation:** the union of the work intervals, divided by the real step time.
- **How to read a trace:** CPU launching above, GPU work below, gaps = idle.
- **The annotation pitfall:** a `record_function` span includes the gaps.

**Tools for this level, in code:**
```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        with record_function("forward"):      # names a block in the trace
            ...
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))   # top operations by GPU time
prof.export_chrome_trace("trace.json")                                      # open in https://ui.perfetto.dev
work = [e for e in prof.events() if e.device_type.name == "CUDA" and not e.is_user_annotation]
```
- PyTorch *Profiler* recipe: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- `torch.profiler` reference: https://docs.pytorch.org/docs/stable/profiler.html
- Perfetto (the trace viewer) docs: https://perfetto.dev/docs/
- PyTorch, *Understanding CUDA Memory Usage*, the memory snapshot tool: https://docs.pytorch.org/docs/stable/torch_cuda_memory.html

**Do:**
21. **Top operations.** Profile 5 training steps and print `key_averages().table(...)`. Which operations take the most GPU time? Are they matrix multiplications or element-wise ops?
22. **Utilisation.** Compute GPU busy time ÷ step time at batch 8 and 32. Then reproduce the 146% bug on purpose by counting annotation events, and fix it.
23. **Read a trace.** Open a trace in Perfetto. Find one forward pass, measure the gaps between its GPU kernels, and find where the CPU is launching the *next* step while the GPU is still busy (the overlap).
24. **Memory snapshot.** Record a memory snapshot of one step (`torch.cuda.memory._record_memory_history`, then dump it, as in the memory docs). Find where the peak happens: forward, backward or optimizer step.

**Check yourself:** why is a sync-free step's GPU utilisation high on the T4 at batch 32 (97%) but low on the RTX 3050 at batch 8 (67%)?

---

## Level 7: More than one GPU

**Understand:**
- **DDP:**
  - one process per GPU, each holding a full copy of the model;
  - gradients averaged by **all-reduce** every step, in buckets that overlap the backward;
  - weights broadcast from rank 0 at the start.
- **The `split_batches` choice:** splitting each batch between the GPUs vs giving each GPU its own batch (the effective batch size changes).
- **The cost model:** the forward and backward are split, but every GPU still does the full optimizer and clip work, plus the gradient exchange (~0.74 GB per step for 184M fp32 gradients).
- **The alternative:** independent runs, one per GPU, with no communication.
- **Accelerate:** a convenience layer over autocast, GradScaler, DDP and process launching.

**Read:**
- Li et al. 2020, *PyTorch Distributed: Experiences on Accelerating Data Parallel Training*: https://arxiv.org/abs/2006.15704
- PyTorch *DDP notes*: https://docs.pytorch.org/docs/stable/notes/ddp.html
- PyTorch, *Getting Started with Distributed Data Parallel*: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
- `torch.distributed` reference (collectives: all-reduce, broadcast, barrier): https://docs.pytorch.org/docs/stable/distributed.html
- Andrew Gibiansky, *Bringing HPC Techniques to Deep Learning*, ring all-reduce explained step by step: https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/
- NCCL, *Collective Operations*: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

**Do:**
25. **All-reduce by hand.** Start 2 CPU processes with `torchrun --nproc_per_node 2` (backend `gloo`). Have each make a tensor, all-reduce it, and print it. Then time all-reduce for tensor sizes from 1 MB to 1 GB.
26. **Estimate the exchange.** From exercise 25's speed (and, on Kaggle **[T4]**, the real two-T4 speed), predict the all-reduce time for DeBERTa's gradients, and compare with the step-time estimate in this folder's discussion of two GPUs.
27. **Same update, two ways.** With DDP on 2 processes and `split_batches`-style halves, check that the averaged gradient equals the gradient of the full batch on one process.
28. **[T4] Two independent runs.** Run two training processes at once, one per GPU (`CUDA_VISIBLE_DEVICES=0` / `1`), and compare each one's time per step with a run alone. Does the shared CPU slow them down?

**Check yourself:**
- Why does splitting a batch of 32 over 2 GPUs not halve the step time?
- When is DDP clearly worth it, and when are independent runs better?

---

## Level 8: Capstone and writing it up

29. **Repeat the whole story on your own.** Take a smaller model (e.g. a BERT-sized transformer from a config), measure fp32 → fp16 → fused AdamW one change at a time, predict each result first, profile the final step, and write a note in the style of note 05.
30. **Write the post.** Three parts:
   - **the training loop as computation:** what each line costs and where it runs;
   - **the three regimes and how to test which one you're in;**
   - **the story:** five changes at once gave nothing; one at a time gave 2.8×, and the profiler explained why the last change did nothing.

---

## Quick reference: the tools

| Tool | Answers | Level |
|---|---|---|
| `torch.cuda.synchronize()` + `time.perf_counter()` | how long did this GPU work take (wall time)? | 1, 3 |
| `torch.cuda.Event(enable_timing=True)` | how long did it take on the GPU's own clock? | 3 |
| `torch.cuda.max_memory_allocated()` / `reset_peak_memory_stats()` | what was the peak memory? | 3 |
| `nvidia-smi` | is the GPU busy, how much memory is used, right now? | 3 |
| `torch.profiler.profile` + `key_averages().table()` | which operations take the time? | 6 |
| `record_function("name")` | which phase does this time belong to? (spans include gaps) | 6 |
| `export_chrome_trace` + Perfetto | what does the CPU/GPU timeline look like, where are the gaps? | 6 |
| memory snapshot (`torch.cuda.memory`) | what is in memory at the peak? | 6 |
| reading the source (`torch/optim/adam.py`, `torch/amp/grad_scaler.py`) | what does this call *really* do (passes, syncs, defaults)? | 4, 5 |
