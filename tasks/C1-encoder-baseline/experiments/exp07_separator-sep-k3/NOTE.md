# exp07: separator [SEP] between segments, k = 3

## Question
Does a real boundary between segments help? exp04 (k = 3) joined segments with `"\n"`, which the tokenizer turns into a plain space: the model saw one run of words with only `[T] … [/T]` around the target. With `[SEP]` between segments, how does val macro-F1 (6) compare with exp04 (0.5524 ± 0.0103)?

## Setup (change vs exp04)
- **Same as exp04:** DeBERTa-v3-base, k = 3, lr 1e-5, 3 epochs, batch 32, warmup 10 %, weight decay 0.01, clipping 1.0, plain cross-entropy, fp16 + fused AdamW, no gradient checkpointing, best epoch by val macro-F1, `[T] … [/T]` markers, context clipped at meeting boundaries.
- **Changed:** separator `" [SEP] "` instead of `"\n"`.
- **Seeds:** 42, 43, 44, compared seed by seed with exp04.
- **How the tokenizer handles it** (checked locally):
  - `" [SEP] "`, `"[SEP]"` and `" [SEP]"` all give the same tokens: `[SEP]` is kept as one special token, id 2.
  - It's the same token the tokenizer already puts at the end of every input. So the model can't tell "end of a segment" from "end of the input" by the token itself, only by position.
  - The boundary carries no speaker information: a change of speaker and the same speaker continuing look the same.
- **Input size, measured locally with the real tokenizer** (batch 32, seed-42 order):

  | separator | padded length per batch: mean / p99 / max | total tokens per epoch | `[SEP]` per input | truncated |
  |---|---|---|---|---|
  | `"\n"` (exp04) | 81 / 215 / 264 (val max 270) | 6.36 M | 1 (the final one) | 0 |
  | `" [SEP] "` | 87 / 221 / 270 (val max 276) | 6.84 M (+7.5 %) | 6.99 on average, at most 7 | 0 |

  Almost every input has its full 3 + 3 context; only inputs near a meeting's start or end have fewer.
- **Memory:** the longest training batch is 32 × 270, against exp04's 32 × 264 that ran without problems. The notebook checks it with one real step first.
- **Expected time:** ~5 % more than exp04, ~32–33 min per run.

## How it was run
- **Notebook:** `notebooks/kaggle_exp07_separator_sep_k3.ipynb` (Kaggle, GPU T4 × 2, two runs at a time). Branch `C1/encoder-baseline`, commit: _fill in from the `git log` line of the "Code and packages" cell_.
- **Memory check:** _peak allocated / reserved, from section 5_.
- **Command per seed:** `CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src python scripts/trial_run.py --mode flat --seeds <seed> --lr 1e-5 --epochs 3 --past-k 3 --future-k 3 --separator " [SEP] " --name exp07_k3_sep --data-root <DATA_ROOT> --model microsoft/deberta-v3-base --out-dir /kaggle/working/runs`
- **Run folders:** `exp07_k3_sep_seed42`, `exp07_k3_sep_seed43`, `exp07_k3_sep_seed44`.

## Decision rule
_Fill in before the runs:_
- what difference from exp04 counts as `[SEP]` helping, hurting, or no different;
- which separator the later k experiments use if they are within seed noise.

## Result
_Pending._

## Conclusion
_Pending._
