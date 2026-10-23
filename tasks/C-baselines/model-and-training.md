# The model and training functions (`src/smi/models/model.py`)

What each function does, the logic step by step, and the library APIs it relies on. The companion of [input-pipeline.md](input-pipeline.md): that document ends with model-ready batches, and this one starts there. Design decisions are in [DESIGN.md](DESIGN.md), Parts 3–4; speed measurements are in [concepts/performance-analysis-and-optimization](../../concepts/performance-analysis-and-optimization/README.md).

```
load_model_and_tokenizer ──► model + tokenizer
build_optimizer(model) ────► AdamW
build_scheduler(optimizer, train_loader) ──► LR schedule
train(model, loaders, optimizer, scheduler) ──► best epoch , its val predictions , history
        └── every epoch: predict(model, val_loader) ──► macro_f1_over ──► keep the best epoch
```

---

## 1. `load_model_and_tokenizer(name_or_path, labels=LABELS)`

**Job:** load a pretrained checkpoint as a classifier over `labels`, together with its own tokenizer.

**The two Hugging Face APIs it uses:**
- **`AutoTokenizer.from_pretrained(name_or_path)`:** reads the checkpoint's tokenizer files and returns the matching tokenizer class (DeBERTa-v3: a SentencePiece tokenizer). "Auto" means you don't name the class: it's chosen from the checkpoint's config.
- **`AutoModelForSequenceClassification.from_pretrained(name_or_path, num_labels, id2label, label2id, dtype)`:** reads the architecture from `config.json`, builds it, loads the pretrained weights, and **replaces the pretraining head with a new, randomly initialised classification head** of `num_labels` outputs. For an encoder like DeBERTa, the head reads the `[CLS]` position; for a decoder, the last real token (DESIGN.md Part 5).
- **Docs:**
  - Auto classes: https://huggingface.co/docs/transformers/model_doc/auto
  - Loading models (`from_pretrained`): https://huggingface.co/docs/transformers/main_classes/model
  - Tokenizers: https://huggingface.co/docs/transformers/fast_tokenizers

**Logic:**
1. **Build `id2label` and `label2id` from `labels`,** so class `i` of the model is `labels[i]`. With the default 7 `LABELS`, that's the same index the Dataset uses, so the two can't disagree.
2. **Load the tokenizer** from `name_or_path`.
3. **Load the model** from the **same** `name_or_path`, so model and tokenizer can never mismatch, with **`dtype=torch.float32` passed explicitly.**
4. **Return** `(model, tokenizer)`.

**Why `dtype` is explicit:** `transformers` 5 defaults to `dtype="auto"`, meaning "load in the precision the checkpoint was stored in". DeBERTa-v3's checkpoint is stored in fp16, so without this the model would load in fp16, and training with fp16 master weights loses small updates. fp16 is applied only during computation, by `train(fp16=True)`.

**The load report printed on every load is expected:**
- **UNEXPECTED:** the checkpoint's pretraining heads (`lm_predictions.*`, `mask_predictions.*`). They're dropped.
- **MISSING:** the new head (`pooler.*`, `classifier.*`). It's created randomly and trained by fine-tuning.

**`name_or_path`** can be a Hub name (downloaded into the Hugging Face cache once) or a local folder, e.g. an attached Kaggle dataset, for runs without internet.

---

## 2. `predict(model, dataloader, labels=LABELS, fp16=False)`

**Job:** run the model over every batch of a split and return one prediction per segment, in the harness's shape.

**Logic:**
1. **`model.eval()`:** dropout off, so predictions are deterministic. Switching back to training mode is the training loop's job.
2. **Find the device the model is already on.** `predict` moves only the batches, never the model.
3. **Under `torch.no_grad()`** (no gradients needed, so less memory and time), for each batch:
   1. Move `input_ids` and `attention_mask` to the device.
   2. **The forward pass**, under fp16 autocast if `fp16=True`.
   3. **Softmax in fp32** (`logits.float().softmax(-1)`), so the scores sum to 1 within the harness's tolerance even if the logits are fp16.
   4. For each row: a score per label (keyed by `labels`), and the **label = the class with the highest score.**
4. **Return** `{seg_id: {"label": ..., "scores": {...}}}`. That's exactly the shape `load_predictions` returns, so the metric functions and `write_predictions` take it directly. Keyed by `seg_id`, so the eval order (longest first) doesn't matter.

---

## 3. The selection metric: `macro_f1_over` and `classification_report`

- **`macro_f1_over(records, gold, select_labels)`:** the mean F1 over `select_labels`, scored only on the seg_ids in `gold`. It's what `train` uses to pick the best epoch.
- **`classification_report(records, gold, labels, select_labels)`:** precision / recall / F1 / support per class plus a confusion matrix, for any label set. Used for the hierarchical model's per-stage reports.

Both reuse the harness's own `prf_from_sets` and sum with Python's `sum()`, like the harness does. With the 7 labels they are **bit-identical** to `macro_f1_6` and `per_class` in `metrics.json`, so the number that selects the epoch is the number that gets reported.

---

## 4. `build_optimizer(model, lr, weight_decay, fused=False)`

**Job:** an AdamW optimizer over the model's trainable parameters, with weight decay on the right ones.

**Why AdamW and not "Adam + L2 regularization":**
- **Adam + L2** adds `weight_decay · w` to the **gradient**. Adam then divides every gradient by `√v`, the running size of that parameter's gradients. So the regularization is divided too: parameters with large gradients get **less** weight decay, those with small gradients more. The amount of regularization ends up depending on the gradient history, which isn't what weight decay is meant to do.
- **AdamW** applies weight decay **separately**, directly to the weights: `w ← w − lr · weight_decay · w`, outside the adaptive scaling. Every parameter shrinks at the same rate. This is "decoupled weight decay" (Loshchilov & Hutter), and it generalises better.
- Paper: https://arxiv.org/abs/1711.05101 · PyTorch docs: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html

**Why biases and LayerNorm weights get no weight decay:**
- **Weight decay pulls parameters towards 0.** That's useful for weight matrices: it discourages large weights and overfitting.
- **A bias is an offset, and a LayerNorm weight is a scale** (it starts at 1). Pulling a LayerNorm scale towards 0 shrinks the normalised activations themselves, and pulling biases towards 0 removes offsets the model needs.
- **They're also tiny in number,** so they contribute almost nothing to overfitting.
- Excluding them is the standard practice for transformers (BERT and its successors).

**Why "fewer than 2 dimensions" selects them:**
- **Biases and norm weights are 1-D vectors** (e.g. shape `[768]`).
- **Weight matrices and embedding tables are 2-D** (e.g. `[768, 768]`, `[128100, 768]`).
- So splitting **by shape** separates exactly these groups. It also works for any architecture without knowing parameter names: decoders name their norms `norm` or `RMSNorm`, not `LayerNorm`, so a name-based rule would break.

**Logic:**
1. **For each parameter,** skip it if it's frozen (`requires_grad=False`, e.g. the base model under LoRA later). Otherwise put it in the **decay** group (2-D or more) or the **no-decay** group (1-D).
2. **Build `torch.optim.AdamW`** with the two groups (`weight_decay` vs `0.0`), the learning rate `lr`, and PyTorch's default betas and eps.
3. **`fused=True`:** the whole update runs in a few large GPU kernels instead of one pass per line of the update. Measured on a T4: 63.5 → 24.0 ms per step. The parameters must already be on the GPU, so move the model first.

**Pitfall (found while adding the option):** `AdamW(..., fused=False)` is **not** the default. An explicit `False` also prevents PyTorch from choosing its default "foreach" implementation, so it falls back to the slow single-tensor loop, which does the arithmetic in a slightly different order. So when not fusing, the function passes `fused=None`, which keeps the default.

---

## 5. `build_scheduler(optimizer, train_loader, epochs, warmup_ratio, grad_accum_steps, scheduler_name="linear")`

**Job:** the learning-rate schedule for the whole run.

**The schedule ("linear" with warmup):**
```
LR
max ┤      ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
    │    ╱                                 ╲
  0 ┼──╱───────────────────────────────────────╲──►  optimizer steps
       └ warmup ┘└────────── linear decay to 0 ─────┘
```
- **Warmup** (the first `warmup_ratio` of the steps, e.g. 10%): the LR rises linearly from 0 to `lr`. At the start, the new classification head is random and Adam's statistics (`m`, `v`) are still empty, so large steps then could damage the pretrained weights.
- **Decay:** the LR then falls linearly to 0 at the last step, so training settles.

**The API:** `transformers.get_scheduler(name, optimizer, num_warmup_steps, num_training_steps)` returns a PyTorch LR scheduler for a schedule chosen **by name** (`"linear"`, `"cosine"`, …), so another schedule becomes a setting. Docs: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules

**Logic:**
1. **Check** that `epochs` and `grad_accum_steps` are ≥ 1 and `warmup_ratio` is in [0, 1).
2. **Count optimizer steps, not batches:** `steps_per_epoch = ceil(len(train_loader) / grad_accum_steps)`, and `total = epochs × steps_per_epoch`.
3. **`warmup = int(warmup_ratio × total)`.**
4. **Return** `get_scheduler(scheduler_name, optimizer, warmup, total)`.

**The contract with the training loop:**
- call `scheduler.step()` after **every** optimizer step, not once per epoch;
- with gradient accumulation, also take an optimizer step on the last, incomplete group of an epoch, because the `ceil` counts that step.

Verified: the executed steps equal the planned total, and the LR ends at 0.

---

## 6. `train(model, train_loader, val_loader, val_gold, optimizer, scheduler, epochs, grad_accum_steps, max_grad_norm, log_every, labels=LABELS, select_labels=TARGET_LABELS, label_map=None, on_log=None, fp16=False)`

**Job:** fine-tune, score val after every epoch, and keep the best epoch.

**Setup:**
1. **Pick the device** (the GPU if there is one) and **move the model there.** `fp16=True` without a GPU raises an error.
2. **Create the loss** (plain cross-entropy) and the fp16 **`GradScaler`.** The scaler does nothing when `fp16=False`.
3. **If `label_map` is given** (the hierarchical stages), build a lookup tensor that turns the Dataset's 7-class label index into the model's own class index (or `-100` = ignored).

**Every epoch:**
1. **`model.train()`** (dropout on) and `optimizer.zero_grad()`.
2. **For every batch:**
   1. **Move the tensors to the device;** map the labels if `label_map` is set.
   2. **The forward pass**, under fp16 autocast if `fp16=True`, then the loss on fp32 logits.
   3. **The backward pass:** the loss, divided by its accumulation group's size, then scaled by the GradScaler, then `.backward()`. Gradients **add up** across the batches of a group.
   4. **Record the loss** with `loss.item()`, for the history.
   5. **At the end of each accumulation group:**
      - unscale the gradients (fp16), then clip their norm to `max_grad_norm`;
      - read the LR this step uses;
      - take the optimizer step through the scaler, which skips it if an fp16 gradient overflowed;
      - update the scaler, then the scheduler;
      - zero the gradients and count the step.
      - **Every `log_every` steps:** write a `step` line (`step`, `epoch`, `train_loss`, `lr`, `grad_norm`, `loss_scale`, `seconds`) and hand it to `on_log`.
3. **After the epoch:**
   1. **`predict` on val** (with `fp16` like training);
   2. **`macro_f1_over`** on `val_gold` over `select_labels`;
   3. **If strictly better** than every earlier epoch (so a tie keeps the earlier one), remember this epoch and **its val predictions**;
   4. **Write an `epoch` line** (`epoch`, `train_loss`, `val_macro_f1`, `per_class_f1`, `is_best`, `seconds`) and hand it to `on_log`.

**Return:** `(selected_epoch, best_val_records, history)`. No weights are saved: after `train` the model holds the **last** epoch's weights, and the best epoch's val predictions are returned instead, which is enough for validation results (DESIGN.md Part 4: checkpoints are designed, not built).

**Options and their defaults:**
- **`fp16=False`:** every scaler call is a no-op, and the step is exactly the fp32 loop. Verified byte for byte against the run made before the option existed.
- **`labels` / `select_labels` / `label_map`:** the same loop trains the 7-class flat model and both hierarchical stages.
- **`on_log`:** the caller decides what happens with each history line. `trial_run.py` prints it and appends it to `history.jsonl` at once.
