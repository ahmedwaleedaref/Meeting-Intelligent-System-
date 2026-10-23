# 05 Measured on the T4, one change at a time

The baseline training step (`train()` in commit `bc45255`: DeBERTa-v3-base, k = 0, real train batches) timed on one Kaggle Tesla T4. Each row adds **one** change to the previous one. Notebook: `notebooks/kaggle_batch_size_test.ipynb`, which says how to measure. Exploration measurements (PROTOCOL §8), not reportable results.

## Time per step and per epoch

| setup | step, batch 32 | step, batch 64 | min/epoch, batch 32 | min/epoch, batch 64 | peak VRAM 32 / 64 |
|---|---|---|---|---|---|
| fp32, default AdamW (the original) | 245.8 ms | 464.6 ms | 10.1 | 9.5 | 7.5 / 13.8 GB |
| + fp16 (autocast + loss scaling) | 165.7 ms | 227.4 ms | 6.8 | 4.7 | 5.5 / 9.3 GB |
| + fused AdamW | **119.3 ms** | **175.4 ms** | **4.9** | **3.6** | 5.5 / 9.3 GB |

Batch 128 does not fit in 15 GB in any of these setups (fp16: ~3.8 GB more per 32 segments → ~17 GB).

## Time per phase (breakdown mode, ms)

| setup | batch | data | forward | backward | clip | optimizer |
|---|---|---|---|---|---|---|
| fp32 | 32 | 2.9 | 69.0 | 121.9 | 10.5 | 62.7 |
| fp32 | 64 | 4.5 | 148.2 | 225.5 | 10.6 | 62.4 |
| fp16 | 32 | 3.0 | 36.8 | 49.9 | 16.3 | 63.5 |
| fp16 | 64 | 4.7 | 40.6 | 92.5 | 16.1 | 63.3 |
| fp16 + fused | 32 | 2.8 | 34.0 | 47.6 | 16.3 | 24.0 |
| fp16 + fused | 64 | 4.2 | 35.9 | 83.6 | 16.2 | 23.9 |

## What each measurement showed

**fp32: compute-bound.** From batch 32 to 64 the tokens grew 2.40× and the step 1.89×; forward 2.15×, backward 1.85×. Time grows almost in proportion to the work, so the arithmetic is the main cost. Backward is ~1.8× forward, as expected (two matrix multiplications per forward one).

**The optimizer and clip are fixed costs.** They're constant across batch sizes because the gradient has one value per weight whatever the batch. The default AdamW alone is 25% of the fp32 step.

**fp16: forward + backward 2.2× faster** (190.9 → 86.7 ms, predicted 70–100). The optimizer is unchanged: it still works on fp32 weights and states. Clip grows by ~6 ms: it now includes the scaler's unscale pass over all gradients. Peak memory drops (activations stored at half size).

**fp16 exposes the overhead floor of the forward.** Forward 36.8 → 40.6 ms for 2.4× the tokens: it barely grows. What's left at batch 32 is mostly fixed cost (kernel launches, the model's Python code). That's why batch 64 pays off much more in fp16 (6.8 → 4.7 min/epoch) than in fp32 (10.1 → 9.5).

**Fused AdamW: 63.5 → 24.0 ms (2.6×).**
- **The prediction from counting memory passes in the default implementation's code:** ~22 passes over the parameters ≈ 16 GB of VRAM traffic, against 7 passes ≈ 5.2 GB fused, so a 3.1× reduction.
- **If 63.5 ms moves 16 GB,** the T4 achieves ~250 GB/s in practice. At that rate 5.2 GB takes ~21 ms; measured: 24 ms.
- **So the traffic estimate explains the result to within ~15%.** This is timing evidence; a direct byte count needs Nsight Compute.
- **Peak memory did not change.** The peak happens during forward/backward, while activations are alive. The optimizer's temporaries only exist during `optimizer.step()`, after the activations are freed.

## Corrections to earlier hypotheses
- **"The step is launch-bound" (report §6, first version) was wrong for the original setup.** In fp32 it is mostly compute-bound. The overhead floor only appears once fp16 has made the computation cheap.
- **The 2-GPU run's missing speed-up is explained by the combination,** not by a single cause:
  - each GPU got 16 segments, which in fp16 sit at the forward's overhead floor;
  - the gradients (~0.74 GB) had to be exchanged every step;
  - fp16's gain was cancelled.

## Synchronisation points found in the code
- **`loss.item()` every batch** (baseline): one sync per step.
- **`GradScaler.step()` with the default AdamW:** checks for overflow with `.item()`, one sync per step (`torch/amp/grad_scaler.py`, `_maybe_opt_step`).
- **`GradScaler.step()` with fused AdamW: no sync.** The optimizer declares `_step_supports_amp_scaling`, so the scaler passes the overflow flag as a GPU tensor and the fused kernel skips the update itself.
- **`scaler.get_scale()` is `scale.item()`, a sync.** The first fp16 measurements called it twice per step to count skipped steps, an artifact of the measurement code, so those fp16 numbers are slightly pessimistic. The counter now runs only in breakdown mode, which syncs anyway.

## No `.item()` per batch: no effect, and why

Measured in one session against fp16 + fused: 117.4 vs 117.9 ms (batch 32), 171.7 vs 173.6 ms (batch 64). The difference is within noise. (A first breakdown table showed backward time "moving" into clip: the breakdown mode had no sync after the backward and relied on `.item()` for it. Fixed.)

**The profiler explains it** (`torch.profiler`, 10 steps; GPU busy time = union of kernel / memset / copy intervals):

| batch | real ms/step | GPU busy | GPU idle | utilisation |
|---|---|---|---|---|
| 32 | 119.8 | 116.0 | 3.9 | **96.8%** |
| 64 | 189.6 | 178.8 | 10.8 | **94.3%** |

- **The GPU is already busy ~95–97% of every step,** so there is no waiting left for a removed sync to recover.
- **The forward on its own is CPU-bound,** with 24.5 ms of GPU work against ~34 ms of launching. But without per-phase syncs, the CPU launches the next forward while the GPU still runs the backward, clip and optimizer (~91 ms of GPU work). The overhead is **hidden by overlap**.
- **On the RTX 3050 at batch 8** the same step showed only **67%** utilisation. The regime depends on the GPU, the batch size and the precision, not only on the model.
- **Pitfall:** `record_function` labels create GPU-side annotation events that span their whole phase, gaps included. Counting them as work gave an impossible 146% utilisation. Count only real work (`not event.is_user_annotation`).

## Conclusion
fp32 / batch 32 / default AdamW → **fp16 / batch 64 / fused AdamW**: 10.1 → 3.6 min per epoch (~2.8×). On one T4 the step is now GPU-bound, so further speed would need **less GPU work** (e.g. kernel fusion with `torch.compile`), not fewer syncs. Batch 64 still needs a validation run to confirm accuracy before it is adopted.
