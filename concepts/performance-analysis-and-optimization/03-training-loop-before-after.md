# 03 The training loop before and after the optimizations

**Before:** commit `bc45255` (the first Kaggle run: fp32, one T4, ~0.27 s per step).
**After:** commit `6e1d913` (fp16, two T4s, same time per step).

All five changes were made together and measured only together, so the run cannot say which helped. See [02](02-performance-regimes.md) for what local measurements suggest.

## What did NOT change
The algorithm. For one batch:
1. forward pass;
2. cross-entropy loss;
3. backward pass (divided by the accumulation group size);
4. clip the gradient norm;
5. AdamW step and LR scheduler step;
6. zero the gradients.

After every epoch: predict on val, then select the best epoch. The same seed reproduced the same loss curve (1.8723 vs 1.8775 at step 50, 0.6989 vs 0.7032 at step 200).

## The five changes, line by line

### 1. Fused AdamW (`build_optimizer`)
```python
# before                                          # after
torch.optim.AdamW(groups, lr=lr)                  torch.optim.AdamW(groups, lr=lr, fused=all_on_gpu)
```
The default implementation ("foreach") updates the parameters in groups of tensors, several kernels per group. Fused does the whole update (m, v and w) for all parameters in a **few large kernels**. That's far fewer launches, and each number is read and written once instead of once per sub-operation. The update itself is memory-bound, since it touches w, g, m and v for 184M parameters every step. It requires the parameters to be on the GPU when the optimizer is built, so the script now calls `model.to(device)` first.

### 2. No `.item()` on every batch (`train`)
```python
# before                                          # after
epoch_loss_sum = 0.0                              epoch_loss_sum = torch.zeros((), device=device)
epoch_loss_sum = epoch_loss_sum + loss.item()     epoch_loss_sum = epoch_loss_sum + loss.detach()
```
Before, every batch had a synchronisation point (see [01](01-cpu-gpu-execution.md)): the CPU waited for the GPU to finish the backward pass before it could tokenize the next batch. Now the sum stays on the GPU and is read only when a line is logged (every 50 steps) and at the end of the epoch.

### 3. Eval batch 128 instead of 32 (`trial_run.py`)
```python
val_loader = build_dataloader(..., batch_size=SETTINGS["eval_batch_size"], ...)   # 128
```
Validation has no backward pass and no optimizer, so it has the memory to spare. 4× more segments per kernel means 4× fewer forward passes for the same 16,188 segments. This is the one change with a visible effect: an epoch including validation went from 11.9 to 11.2 min.

### 4. fp16 mixed precision (Accelerate)
```python
# before                                          # after
loss.backward()                                   accelerator.backward(loss)        # loss scaling inside
torch.nn.utils.clip_grad_norm_(...)               accelerator.clip_grad_norm_(...)  # unscales first
logits = model(...).logits                        logits = model(...).logits        # the prepared model runs under fp16 autocast
loss = loss_function(logits, labels)              loss = loss_function(logits.float(), labels)
```
The matrix multiplications run in fp16 on Tensor Cores. The weights stay fp32 ("master weights"). The loss is multiplied by a scale factor before the backward pass, so small fp16 gradients don't round to zero, and the gradients are divided by it again before clipping and the update. If a gradient overflowed, the step is **skipped** and the scale is lowered (`skipped_steps` in the history: 0 in every run). Checking for overflow needs one synchronisation per step.

### 5. Two GPUs (Accelerate, `split_batches=True`)
```python
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
...
with accelerator.no_sync(model): ...                # inside an accumulation group: no gradient exchange yet
mean_loss = accelerator.reduce(log_loss_sum, ...)   # average the logged loss over both GPUs
accelerator.wait_for_everyone()                     # barrier around validation (main GPU only)
```
What the two GPUs do in each step is in [04](04-data-parallel-two-gpus.md).

## Also changed, but not for speed
- The sampler puts the short batch last (needed by change 5).
- Label maps and `select_labels` (for the hierarchical model).
- Live history lines.
