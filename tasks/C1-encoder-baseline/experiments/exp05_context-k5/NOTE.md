# exp05: context k = 5, separator "\n"

## Question
Does more context help further? With 5 segments before and 5 after the target, how does val macro-F1 (6) compare with k = 3 (exp04: 0.5524 ± 0.0103) and k = 0 (exp03: 0.5225 ± 0.0059)?

## Setup (change vs exp04)
- **Same as exp04:** DeBERTa-v3-base, lr 1e-5, 3 epochs, batch 32, warmup 10 %, weight decay 0.01, clipping 1.0, plain cross-entropy, fp16 + fused AdamW, best epoch by val macro-F1, separator `"\n"` (no boundary between segments, only `[T] … [/T]` around the target), context clipped at meeting boundaries.
- **Changed:** `past_k = future_k = 5`.
- **Seeds:** 42, 43, 44.
- **Input size, measured locally with the real tokenizer** (batch 32, seed-42 order):

  | | padded length per batch: mean / p99 / max | total tokens per epoch | inputs > 512 tokens |
  |---|---|---|---|
  | k = 3 | 81 / 215 / 264 (val max 270) | 6.4 M | 0 |
  | k = 5 | 119 / 289 / 327 (val max 357) | 9.4 M (1.5× k = 3) | 0 |

- **Memory:** the longest training batch is 32 × 327.
  - Measured locally (fp16 training step, random-weight DeBERTa-v3-base): ~292 MB per sequence at L = 264, ~381 MB at L = 327.
  - Estimate at batch 32: ~14.7 GB, against ~12 GB for k = 3. Borderline on a T4.
  - **Rule, decided before the run:** plain batch 32, as in exp04, if the notebook's memory check (one real step on the longest batch) peaks at ≤ 13.5 GB reserved. Otherwise gradient checkpointing (`--grad-checkpointing`): the same batches and the same gradients (checked locally, bit for bit, dropout included), ~1/3 more time per training step.
- **Expected time:** ~13–14 min per epoch, ~40–45 min per run (k = 3: ~10.3 and ~31); ~1/3 more with checkpointing.

## How it was run
- **Notebook:** `notebooks/kaggle_exp05_context_k5.ipynb` (Kaggle, GPU T4 × 2, two runs at a time). Branch `C1/encoder-baseline`, commit: _fill in from the `git log` line of the "Code and packages" cell_.
- **Memory check:** _peak allocated / reserved, from section 5_. Gradient checkpointing: _on / off_.
- **Command per seed:** `CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src python scripts/trial_run.py --mode flat --seeds <seed> --lr 1e-5 --epochs 3 --past-k 5 --future-k 5 [--grad-checkpointing] --name exp05_k5 --data-root <DATA_ROOT> --model microsoft/deberta-v3-base --out-dir /kaggle/working/runs`
- **Run folders:** `exp05_k5_seed42`, `exp05_k5_seed43`, `exp05_k5_seed44`.

## Decision rule
_Fill in before the runs:_ what difference from k = 3 counts as more context helping, hurting, or no different.

## Result
_Pending._

## Conclusion
_Pending._
