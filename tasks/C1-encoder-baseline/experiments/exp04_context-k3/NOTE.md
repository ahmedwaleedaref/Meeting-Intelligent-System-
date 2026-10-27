# exp04: context k = 3, separator "\n"

## Question
Does context help? With 3 segments before and 3 after the target (k = 3), how does val macro-F1 (6) compare with k = 0 (exp03: 0.5225 ± 0.0059)?

## Setup (change vs exp03)
- **Same as exp03's adopted cell:** DeBERTa-v3-base, lr 1e-5, 3 epochs, batch 32, warmup 10 %, weight decay 0.01, clipping 1.0, plain cross-entropy, fp16 + fused AdamW, best epoch by val macro-F1.
- **Changed:** `past_k = future_k = 3`.
  - Context is clipped at meeting boundaries.
  - Context segments are joined with `"\n"`, which DeBERTa's tokenizer turns into a plain space. So the model sees no boundary between segments: the only structure is the `[T] … [/T]` markers around the target.
  - The separator gets its own experiment later.
- **Seeds:** 42, 43, 44.
- **Input size, measured locally with the real tokenizer:**

  | | padded length per batch: mean / p99 / max | total tokens per epoch | truncated |
  |---|---|---|---|
  | k = 0 | 21 / 82 / 151 | 1.6 M | 0 |
  | k = 3 | 81 / 216 / 264 (val max 270) | 6.4 M (3.9×) | 0 |

- **Memory:** the longest training batch is 32 × 264 tokens. The estimate from activations measured locally is ~12 GB of the T4's 15 GB. The notebook checks it with one real step before the runs.

## How it was run
- **Notebook:** `notebooks/kaggle_exp04_context_k3.ipynb` (Kaggle, GPU T4 × 2, two runs at a time). Branch `C1/encoder-baseline`, commit: _fill in from the `git log` line of the "Code and packages" cell_.
- **Memory check:** _peak allocated / reserved, from section 5_.
- **Command per seed:** `CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src python scripts/trial_run.py --mode flat --seeds <seed> --lr 1e-5 --epochs 3 --past-k 3 --future-k 3 --name exp04_k3 --data-root <DATA_ROOT> --model microsoft/deberta-v3-base --out-dir /kaggle/working/runs`
- **Run folders:** `exp04_k3_seed42`, `exp04_k3_seed43`, `exp04_k3_seed44`.

## Decision rule
_Fill in before the runs:_
- what difference from k = 0 counts as context helping (or hurting);
- whether k = 5 and k = 10 are run next whatever the result, or only if k = 3 helps.

## Result
_Pending._

## Conclusion
_Pending._
