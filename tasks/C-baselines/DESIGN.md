# Phase C design: config-driven baselines in `src/`

The answers to the questions in [design-task.md](design-task.md), as designed and as built on branch `C/baselines-design`. One code path for three families: **encoder**, **decoder + classification head**, **prompting**. For results and measurements see [REPORT.md](REPORT.md); for the speed work see [concepts/performance-analysis-and-optimization](../../concepts/performance-analysis-and-optimization/README.md).

**Status labels used below**
- **Built:** implemented in `src/` and tested; use it as it is.
- **Designed:** decided, not implemented yet.
- **Open:** not decided yet.

| Part | Status |
|---|---|
| 1 Data and examples | Built |
| 2 Model input | Built for fine-tuned models; prompt rendering designed |
| 3 Model abstraction | Functions built; registry and family interface designed |
| 4 Training | Built, except checkpoints and resume |
| 5 Decoder | Designed |
| 6 Prompting | Designed (API budget open) |
| 7 Config system | Open |
| 8 Orchestration | `evaluate()` built; `run.py` open (a temporary `scripts/trial_run.py` exists) |
| 9 Inputs | Answered |

---

## Part 9: Inputs (answered first, because they shape the rest)

| Input | Answer |
|---|---|
| GPU | Kaggle, 2 × Tesla T4: two **separate** cards of ~15 GB each (they don't pool into 30 GB), no bf16, no FlashAttention-2 |
| GPU budget | ~30 GPU-h per account per week; 12 h maximum per session |
| API access and budget | **Open:** decide before building prompting with an API backend |
| Offline or real-time | **Offline:** future context is allowed |

**Consequences for the design:**
- **Precision:** fp16, not bf16, with fp32 master weights.
- **Memory:** one model must fit on one T4 (15 GB).
- **The 12 h cap:** resume after an interrupted session is needed (Part 4).
- **Storage:** limited, so keep few checkpoints.
- **Code reaches Kaggle by `git clone`,** and the data as a private Kaggle dataset (`data/` is not in git).

---

## Part 1: Data and examples. **Built** (`src/smi/inputs/dataset.py`)

**What one example is:** one target segment, its gold label, and its context window. k = 0 (target only) is the first baseline, and every k runs through the same code.

**Context:**
- **Selected by segment count:** `past_k` segments before and `future_k` after the target, as two separate settings. Past-only (`future_k = 0`) is therefore a free ablation.
- **Planned grid:** k ∈ {0, 1, 3, 5}, symmetric and left-only.
- **Future context is allowed,** since the target system is offline.
- **Never crosses meeting boundaries.**

**Speakers: dropped for now.** Turning them back on (e.g. "relative": same / other speaker than the target) is a later step and requires a change to the Dataset item.

**Selection and formatting are separate steps:**
- **The Dataset decides *which* segments** form the window, and returns plain text.
- **The family's Collator or prompt renderer decides *how* they look** (Part 2).

**The layer is shared by all three families.** It has no tokenizer and does no rendering, which is what makes it shareable.

**Interface (built):**
```python
class SegmentItem(TypedDict):
    seg_id: str              # target id, the join key with the harness
    label: int | None        # index into LABELS ; None for the test split
    left: list[str]          # previous segments , index 0 = CLOSEST (reverse speaking order)
    target_text: str
    right: list[str]         # next segments , index 0 = closest

SegmentDataset(split_dir, past_k, future_k, allow_test=False, keep_labels=None)
```

**Rules:**
- **Meeting files are read in sorted name order,** so index `i` is the same segment on every machine.
- **The test split loads only with `allow_test=True`,** and then has `label=None` (PROTOCOL §7).
- **`keep_labels`** keeps only targets with those gold labels; their context still comes from every segment. The hierarchical model's stage 2 uses it.
- **Items are read-only:** renderers must not modify them (e.g. use `reversed(left)`, not `left.reverse()`).

The context selection is re-implemented here. Task A's `ContextFormatter` (`src/smi/inputs/formatter.py`) still exists; replacing it needs the agreement of task A's owner.

---

## Part 2: Turning examples into model input. **Built for fine-tuned models**

**One shared content, two renderings:**
- **Fine-tuned (encoder and decoder): built.** The segments are joined in speaking order with a separator and the target is wrapped in markers. For the **decoder only**, a fixed suffix is appended at the end (Part 5). The **encoder has no suffix** (`suffix=""`, the default), because its `[CLS]` token already sees the whole input in both directions:
  ```
  separator.join(reversed(left) + [open_marker + target_text + close_marker] + right) + suffix
  ```
- **Prompting: designed.** An instruction plus label definitions, then the same marked window, then a question about the target, wrapped in the model's chat template (Part 6).

**The target is marked with `[T] … [/T]` as plain text,** at every k including k = 0. The tokenizer splits the markers into ordinary pieces, so no new embeddings are needed. Planned experiments: special tokens vs single punctuation markers, and with vs without markers at k = 0. The markers are an argument, so these are settings, not code changes.

**Separator: open.** `"\n"` is used now. **Finding:** DeBERTa-v3's tokenizer turns `"\n"` into an ordinary space, so the model sees no segment boundary. `tokenizer.sep_token` (`[SEP]`) survives as one token. Decide this before context experiments (k > 0).

**Truncation:**
- **Built:** the tokenizer truncates at `max_length`, cutting from the end, and the Collator counts truncated inputs. At k = 0 the count is 0 in every run.
- **Designed, deferred:** dropping whole segments, farthest from the target first. On a tie (same distance), the **future** segment goes first, because in a conversation what came before a turn is usually what it responds to. Never drop the target or the suffix; if the target alone doesn't fit, cut the target's text, not the suffix. `left` and `right` are closest-first, so "the farthest" is always the last element of its list.
- **Measured** (DeBERTa tokenizer, markers and decoder suffix included, every 10th train window): the longest window is 100 tokens at k = 0, 247 at k = 3, 332 at k = 5, and 519 at k = 10 (1 of 7,891 over 512). **For the planned grid (k ≤ 5), truncation never fires.** The rule is a guard for longer contexts and for prompting, whose inputs are much longer.

**The tokenizer belongs to the model family,** loaded from the same checkpoint name as the model. The Dataset never sees it.

**Batching (built):**
- **`LengthBucketBatchSampler`:** batches of similar character length, for little padding.
  - **Train:** shuffle → chunks of 50 batches → sort inside each chunk → batches → shuffle the batch order. Chunking keeps batches mixed: a full sort would group the short backchannels together.
  - **Eval:** sorted longest first.
  - **Batches are built once,** from the sampler's own seeded generator, so every epoch uses the same order, and every model gets the same batches for the same seed.
- **`SegmentCollator`:** renders, tokenizes the batch in one call (`padding="longest"`, `return_token_type_ids=False`), and returns `input_ids [B, L]`, `attention_mask [B, L]`, `labels [B]` (`-100` for test items) and `seg_ids` (a list).
  - **No family branches:** every family difference is a constructor argument (separator, markers, and `suffix`, which only the decoder sets) or the tokenizer passed in.
- **`num_workers = 0`:** batch preparation costs ~3 ms per step. Worker processes would also hold copies of the Collator, and its truncation counter would stay at 0.

**Interfaces (built):**
```python
LengthBucketBatchSampler(dataset, batch_size, chunk_batches, shuffle, seed=0)
SegmentCollator(tokenizer, max_length, separator, markers=("[T]", "[/T]"), suffix="")
build_dataloader(split_dir, tokenizer, past_k, future_k, batch_size, chunk_batches, shuffle, max_length,
                 separator, markers=("[T]", "[/T]"), suffix="", seed=0, allow_test=False, keep_labels=None)
```

---

## Part 3: The model abstraction. **Functions built; registry and interface designed**

**Designed interface:** every family provides three operations, and `run.py` calls only these:

| Operation | Encoder / decoder | Prompting |
|---|---|---|
| **build** (from config) | model + tokenizer + Collator | backend (local or API) + prompt template |
| **fit** (train, val) | training loop, scores val every epoch, keeps the best epoch | does nothing |
| **predict** (data) → seg_ids + scores | forward pass, softmax | prompt each example, read label log-probabilities (or no scores) |

- **The training loop sits behind `fit`,** so prompting, which has no training, fits the same interface.
- **Encoder and decoder are one implementation** (`AutoModelForSequenceClassification`) with decoder settings switched on (Part 5), not two classes.
- **Prompting** uses the same interface, with an empty `fit`.
- **The config selects the family through a registry:** a table in `src/` from a short name (`encoder`, `decoder`, `prompting`) to its class. It's chosen over import paths because refactors don't break old configs, typos fail when the config is loaded, and configs can only reach reviewed code.

**Built so far:** the encoder side as functions, not yet as a family class or a registry. `scripts/trial_run.py` wires them together:
```python
load_model_and_tokenizer(name_or_path, labels=LABELS) -> (model, tokenizer)   # weights loaded in fp32
predict(model, dataloader, labels=LABELS, fp16=False) -> {seg_id: {"label", "scores"}}
train(model, train_loader, val_loader, val_gold, optimizer, scheduler, epochs, grad_accum_steps,
      max_grad_norm, log_every, labels=LABELS, select_labels=TARGET_LABELS, label_map=None,
      on_log=None, fp16=False) -> (selected_epoch, best_val_records, history)
```
- **`labels`, `select_labels` and `label_map`** let the same functions train other label sets. The hierarchical model's two stages use them (`src/smi/models/hierarchical.py`).
- **`load_model_and_tokenizer` passes `dtype=torch.float32` explicitly:** `transformers` 5 would otherwise load DeBERTa-v3 in its stored fp16.

**Predictions are converted to the harness schema in one place,** `write_predictions` (`src/smi/eval/prediction_schema.py`).
- `predict` returns records already in the harness shape, with label = argmax of the scores.
- The writer validates every record, then writes the file, sorted by `seg_id`. It never overwrites an existing file.

---

## Part 4: Training. **Built, except checkpoints and resume** (`src/smi/models/model.py`)

**Own training loop in plain PyTorch.** The design chose Accelerate; the built loop is plain PyTorch, with mixed precision written out explicitly (`torch.autocast` + `GradScaler`). Multi-GPU training (DDP, through Accelerate or `torchrun`) isn't needed with one run per GPU. The reasons for an own loop rather than HF `Trainer`:
- a custom batch sampler;
- `seg_ids` inside batches;
- epoch selection with the harness's code;
- an own history format.

**Class imbalance:**
- **Plain cross-entropy for now,** so the effect of handling minority classes can be measured against a clean baseline. That becomes its own later task.
- **Designed for that task:** class weights ∝ (1/n_c)^α from train counts only, the same scheme for encoder and decoder, the weights recorded with the run.

**Best epoch:**
- **Selected by the highest val macro-F1 over `select_labels`** (default: the 6 target classes), computed with the harness's own `prf_from_sets`. It's bit-identical to `macro_f1_6` in `metrics.json`.
- **Ties keep the earlier epoch.**
- **The selected epoch is recorded** in `metrics.json`, via `evaluate(..., extra={"selected_epoch": ...})`.

**`history.jsonl`:** the loop creates one line per event and hands it to the `on_log` callback, which prints it and appends it to the file at once. A stopped run therefore keeps its history.
- **`step` line** (every `log_every` optimizer steps): `step`, `epoch`, `train_loss`, `lr`, `grad_norm`, `loss_scale` (fp16 only), `seconds`.
- **`epoch` line:** `epoch`, `train_loss`, `val_macro_f1`, `per_class_f1`, `is_best`, `seconds`.

**Other training rules (built):**
- **Gradient accumulation** divides each batch's loss by its own group's size, and the last incomplete group also gets an optimizer step.
- **Gradient clipping** at `max_grad_norm`, after fp16 unscaling.
- **AdamW,** with weight decay on 2-D parameters only (split by tensor shape, so it works for decoders too).
- **Linear warmup, then decay**, counted in optimizer steps.

**Speed options** (measured one at a time on a T4, `concepts/.../05`):
- `train(fp16=True)`: fp16 autocast + loss scaling;
- `build_optimizer(fused=True)`: fused AdamW.

Both default to `False`, and the default path reproduces the earlier fp32 runs byte for byte. `trial_run.py` turns both on; `--fp32` turns them off.

**Checkpoints: designed, not built.**
- **`last`:** the full state (weights, optimizer, scheduler, scaler, random states, epoch), for resuming after the 12 h cap.
- **`best`:** weights only, needed for test predictions.
- **For LoRA,** only the adapter and the head are saved.
- **Writes are atomic** (temporary folder, then rename).
- **Currently** no weights are saved: the best epoch's val predictions are kept in memory, which is enough for validation results. Test predictions will need `best`.

---

## Part 5: Decoder + classification head. **Designed**

**One label per input:** `AutoModelForSequenceClassification` on a decoder (Qwen family) replaces the language-model head with a new 7-class `score` head on the **last real token**. Causal attention means only that token has seen the whole input.

**A fixed suffix,** e.g. `"\nLabel of [T]:"`, so the pooled token is the same in every example and its only job is to summarise the marked target. The Collator already takes it (`suffix=`).
- The suffix is reserved in the truncation budget and never cut.
- The encoder gets no suffix: its `[CLS]` already sees both directions.

**Pad token and padding side:**
- **How the decoder classifier finds the last real token** (`transformers` 5.20, `modeling_layers.py`, shared by all decoder classifiers): it takes the **rightmost token that is not `config.pad_token_id`**. Without a pad id, any batch larger than 1 fails ("Cannot handle batch sizes > 1 if no padding token is defined").
- **So set a pad token on the tokenizer, and the same id on `model.config.pad_token_id`.** The tokenizer writes the padding; the model finds the last real token by looking for that id. (`model.config` is the in-memory config loaded from the checkpoint's `config.json`; setting it changes only the loaded model, not the cached file.)
- **Reusing EOS as the pad token:** any EOS at the *end* of an input then counts as padding and is skipped, so the pooled token becomes the one before it (the suffix's last token). An EOS in the middle is unaffected.
- **Padding side: right.** This version pools correctly with left or right padding, and right padding keeps positions simple (they start at 0).
- **A test before the first run:** for a padded batch, the pooled position must be the last suffix token of every example.
- **Pin `transformers`,** since this logic has changed between versions.
- **Encoders don't need any of this:** DeBERTa's classifier pools `[CLS]` and never reads `pad_token_id`. Its tokenizer pads with `[PAD]` (id 0) on the right, and the `attention_mask` handles the padding.

**LoRA, not full fine-tuning:** full fine-tuning of a 1.5B model needs ~24 GB, more than a T4.
- **The classification head must be fully trainable;** check the trainable-parameter count at the start of every run.
- **LoRA usually needs a higher LR** (~1e-4 to 2e-4) than full fine-tuning.
- **The LoRA settings stay fixed across models,** so a model comparison isn't a LoRA-settings comparison.

**Memory:**
- **Frozen weights:** fp16 is ~2 bytes per parameter, 4-bit (QLoRA) ~0.5 bytes. On one T4: 0.5–1.5B models fit in fp16 with LoRA, 3B is safer in 4-bit, 7B only in 4-bit with gradient checkpointing.
- **Activations** scale with batch × length (so with k), and often dominate.
- **The compute dtype is fp16** (no bf16 on T4). Keep the head and LoRA weights in fp32, and watch the loss scale in the history.

**Designed config fields:**
```yaml
model:
  family: decoder
  name: <checkpoint>
  dtype: fp16
  quantization: none            # none | 4bit
  gradient_checkpointing: false
  lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: [q_proj, k_proj, v_proj, o_proj]}
input:
  suffix: "\nLabel of [T]:"
train:
  lr: 1.0e-4
```

**Truncation must never cut the suffix:** its last token is the one the head reads. The Collator currently truncates from the **end**, which is exactly where the suffix is. At k ≤ 5 nothing is truncated (Part 2), but for decoder runs:
- stop or warn if the Collator's `n_truncated > 0`;
- before using long contexts, implement Part 2's segment-dropping rule, or set the decoder tokenizer's `truncation_side = "left"` as an interim measure.

The decoder can also use a larger `max_length` (e.g. 1024).

**Open:** the suffix wording; the model list and sizes; the LoRA defaults (set in the first decoder experiment, then fixed).

**What the existing code already provides:** the Collator's `suffix`, label maps, shape-based weight decay, and skipping frozen parameters in the optimizer.

**What remains to write:** setting the pad token on tokenizer and model, the LoRA wrapper, optional 4-bit loading, and the pooling test.

---

## Part 6: Prompting. **Designed**

**Backends, behind one interface:** a rendered prompt plus the 7 label tokens go in; generated text, plus label log-probabilities if the backend has them, come out.

| Backend | Scores |
|---|---|
| Local model (e.g. vLLM on a T4, fp16) | yes: one forward pass, read the 7 letter logits |
| Hosted open models (OpenAI-compatible APIs) | yes: `top_logprobs` |
| Frontier APIs without log-probabilities | no: `scores: null` |

**Suggested runs:**
1. **A local model of the same family as the decoder baseline:** prompting vs fine-tuning on the same model.
2. **One larger hosted open model with log-probabilities:** what scale buys.
3. **Optionally, one frontier model, label-only:** an upper-bound reference, if the budget allows.

Any "thinking" mode is off.

**Storage:** templates and label definitions are **files in one shared folder**, never in experiment folders.
- **Immutable:** a changed prompt is a new file.
- **Added through a reviewed PR.**
- **Referred to by name in the config.**
- **A hash of the rendered template goes into the run's metadata.**

**How labels are shown:**
- **Letters plus descriptive names** (`A) suggestion … G) other`). Each answer is a single token, so it can be scored by log-probability; check this per tokenizer.
- **Never the raw MRDA codes.**
- **Definitions** (from the MRDA manual) are an on/off variable.
- **`other` is always defined.**
- **The letter order and the wording are fixed** across runs.

**Scores:**
- **Local:** a softmax over the 7 letter logits only. No generation, so no parsing failures. Log "mass outside the 7" as a prompt-health check.
- **Hosted:** `top_logprobs`, with a floor value for a letter that's missing.
- **No log-probabilities:** `scores: null`. Those runs are comparable only on label metrics (macro-F1).

**Unparseable output** (null-score backends only):
1. constrain the output;
2. a strict parse;
3. one retry;
4. fallback `other`.

`other` is not a target class, so the fallback adds no false positives to the primary metric. The failure rate goes into `metrics.json`.

**Few-shot:**
- **Examples from train only,** 1–2 per class, drawn with a fixed seed.
- **The same examples and order in every cell** of the grid.
- **Their seg_ids are recorded.**
- **At k > 0, the examples get context windows too.**

**Cost:** one call per segment, so 16,188 per val run, ~10–13M input tokens zero-shot.
- **Every response is cached,** keyed by model + rendered prompt + parameters.
- **The API model version and the date are recorded.**
- **Local scoring is deterministic.**

**Experiment grid:** definitions on/off × shots 0 / 1–2 per class × k ∈ {0, 1, 3, 5} = 16 cells, run locally with one fixed model. Only the best 1–2 configurations go to larger models. Run 2–3 paraphrases of the template on the best cell to measure prompt noise.

**What the existing code already provides:** the Dataset and items (Part 1), the harness (`write_predictions` accepts `scores: null` for prompting), and `evaluate()`.

**What remains to write:** the prompt renderer, the backends, the cache, and parsing.

**Open:** the API budget (Part 9).

---

## Part 7: Config system. **Open**

Not designed yet. `scripts/trial_run.py` uses a Python `SETTINGS` dictionary as a temporary stand-in. The questions to answer:
- What file format, and how does an experiment config inherit from a base?
- How are typos and unknown keys handled?
- Where do defaults live, and how is it guaranteed that new options don't change old configs?
- How much of the config is shared across families, and how much is family-specific?

**Known inputs:**
- the settings `trial_run.py` uses today (data, context, input, model, train, fp16, fused);
- the decoder fields above;
- PROTOCOL §3.3: an experiment's `config.yaml` holds the base config path plus only the fields it changes.

---

## Part 8: Orchestration and evaluation. **`evaluate()` built; `run.py` open**

**`evaluate()` (built, `src/smi/eval/evaluate.py`):**
```python
compute_metrics(records, gold) -> dict      # in memory : the per-epoch selection uses the same code
evaluate(pred_path, split_dir, out_path, allow_test=False, extra=None) -> dict   # file -> metrics.json
```
- **`metrics.json` holds:**
  - `split`, `data_version` (from `source.json`), `allow_test`, `n_segments`;
  - `primary.macro_f1_6`, `per_class`, `macro_f1_7`;
  - `confusion` (counts and row-normalised);
  - `pr_auc` (null until its definition is decided);
  - plus the `extra` fields, e.g. `selected_epoch`.
- **It refuses to overwrite.**
- **It raises** on a label that isn't the argmax of its scores.
- **It's called after training,** on the best epoch's prediction file.

**Seeding (built, in `trial_run.py`):** one seed per run, through `transformers.set_seed` (head initialisation, dropout). The same seed goes to the train sampler, which uses its own generator, so it also picks the batches.

**`run.py`: open.** Not designed yet. `scripts/trial_run.py` is a temporary driver, and its runs are trial runs, not protocol runs. The questions:
- What exactly does `run.py` do, and in what order (config → build → fit → predict → write → evaluate → results table)?
- Run IDs, the run folder, `meta.json` (commit, data version, hardware, seed, the batch size actually used).
- Resume after the 12 h cap, together with Part 4's checkpoints.
- How a code checkout without git (e.g. on Kaggle) still records its commit, and how the "clean working tree" rule (PROTOCOL §4.1) is enforced there.
