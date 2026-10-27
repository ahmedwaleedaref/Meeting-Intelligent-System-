# exp06: context k = 10, separator "\n"

## Question
Does a much wider context help, or does it start to dilute the target? With 10 segments before and 10 after, how does val macro-F1 (6) compare with k = 5 (exp05), k = 3 (exp04: 0.5524 ± 0.0103) and k = 0 (exp03: 0.5225 ± 0.0059)?

## Setup (change vs exp04)
- **Same as exp04:** DeBERTa-v3-base, lr 1e-5, 3 epochs, batch 32, warmup 10 %, weight decay 0.01, clipping 1.0, plain cross-entropy, fp16 + fused AdamW, best epoch by val macro-F1, separator `"\n"` (no boundary between segments, only `[T] … [/T]` around the target), context clipped at meeting boundaries.
- **Changed:**
  - `past_k = future_k = 10`;
  - **gradient checkpointing on** (`--grad-checkpointing`): k = 10 doesn't fit at batch 32 without it (see Memory).
- **Seeds:** 42, 43, 44.
- **Input size, measured locally with the real tokenizer** (batch 32, seed-42 order):

  | | padded length per batch: mean / p99 / max | total tokens per epoch | inputs > 512 tokens |
  |---|---|---|---|
  | k = 3 | 81 / 215 / 264 (val max 270) | 6.4 M | 0 |
  | k = 5 | 119 / 289 / 327 (val max 357) | 9.4 M | 0 |
  | k = 10 | 212 / 456 / 512 (val max 512) | 16.7 M (2.6× k = 3) | 1 of 78 903 train (513 tokens), 1 of 16 188 val (521) |

  - The collator cuts from the end, so the 2 long inputs lose a few tokens of future context. The target is never cut (checked: it ends before token 512 in both).
- **Memory:**
  - Measured locally (fp16 training step, random-weight DeBERTa-v3-base): ~700 MB per sequence at L = 512 without checkpointing, ~76 MB with it.
  - Estimate at batch 32: ~25 GB without checkpointing (doesn't fit a T4's ~15), ~5 GB with.
- **Gradient checkpointing doesn't change the run.** Each layer's activations are recomputed in the backward instead of stored.
  - Checked locally: the same loss and gradients, bit for bit, with dropout on (the RNG state is replayed).
  - So the batches, the updates and the selected epoch are what batch 32 without checkpointing would give. Only time and memory change.
  - Cost: ~+33 % per training step (measured locally); evaluation is unaffected.
- **Expected time:** ~25–30 min per epoch, ~80–100 min per run, ~3–3.5 h for the 3 seeds on 2 GPUs.

## How it was run
- **Notebook:** `notebooks/kaggle_exp06_context_k10.ipynb` (Kaggle, GPU T4 × 2, two runs at a time). Branch `C1/encoder-baseline`, commit: _fill in from the `git log` line of the "Code and packages" cell_.
- **Memory check:** _peak allocated / reserved, from section 5_.
- **Command per seed:** `CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src python scripts/trial_run.py --mode flat --seeds <seed> --lr 1e-5 --epochs 3 --past-k 10 --future-k 10 --grad-checkpointing --name exp06_k10 --data-root <DATA_ROOT> --model microsoft/deberta-v3-base --out-dir /kaggle/working/runs`
- **Run folders:** `exp06_k10_seed42`, `exp06_k10_seed43`, `exp06_k10_seed44`.

## Decision rule
_Fill in before the runs:_
- what difference from the best of k = 3 / k = 5 counts as k = 10 helping, hurting, or no different;
- how the best k is picked if two are within seed noise (e.g. the smaller k, since it's ~2–3× cheaper).

## Result
_Pending._

## Conclusion
_Pending._
