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
Not written down before the runs (the same as exp04 and exp05).

## Result
All 3 runs finished, with no out-of-memory error. Each one's `settings.json` matches the setup (separator `" [SEP] "`, `past_k = future_k = 3`, lr 1e-5, 3 epochs, no gradient checkpointing). 0 truncated train inputs, and every seed selected epoch 3. ~10.2 min per epoch on GPU 0 (seeds 42, 44) and ~11.1 on GPU 1 (seed 43): ~31–33 min per run, the same as exp04 (~31).

Val macro-F1 (6), selected epoch in brackets:

| seed | `"\n"` (exp04) | `" [SEP] "` | difference |
|---|---|---|---|
| 42 | 0.5419 (3) | 0.5545 (3) | +0.013 |
| 43 | 0.5624 (3) | 0.5403 (3) | −0.022 |
| 44 | 0.5529 (2) | 0.5539 (3) | +0.001 |
| **mean ± std** | 0.5524 ± 0.0103 | **0.5496 ± 0.0080** | −0.003 ± 0.018 |

Macro-F1 (7): 0.6069 ± 0.0088 → 0.6047 ± 0.0067.

Val macro-F1 (6) per epoch:

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| `"\n"` | 0.425, 0.532, 0.542 | 0.447, 0.559, 0.562 | 0.449, 0.553, 0.549 |
| `" [SEP] "` | 0.423, 0.522, 0.555 | 0.390, 0.463, 0.540 | 0.456, 0.534, 0.554 |

Per-class F1, mean (std) over the 3 seeds:

| | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| `"\n"` | 0.525 (0.019) | 0.465 (0.015) | 0.547 (0.009) | 0.619 (0.006) | 0.705 (0.006) | 0.455 (0.027) | 0.934 (0.000) |
| `" [SEP] "` | 0.538 (0.015) | 0.470 (0.037) | 0.539 (0.005) | 0.620 (0.003) | 0.712 (0.001) | 0.418 (0.047) | 0.935 (0.001) |
| difference | +0.013 | +0.004 | −0.007 | +0.001 | +0.007 | −0.036 | +0.001 |

`cc` per seed with `[SEP]`: 0.468 / 0.374 / 0.414. Precision / recall: 0.480 / 0.432 → 0.576 / 0.337. That's ~26 fewer `cc` predictions per seed, ~8 of them correct.

Confusion, mean over the 3 seeds, the cells that changed by more than 10 (gold → predicted):

| | `"\n"` | `" [SEP] "` |
|---|---|---|
| `aa` → `aa` (of 817) | 446 | 424 |
| `bk` → `aa` | 115 | 103 |
| `other` → `aa` | 247 | 225 |
| `other` → `cs` | 195 | 163 |
| `other` → `cc` | 35 | 21 |
| `other` → `other` | 12704 | 12765 |

Same seed, `"\n"` vs `" [SEP] "`: the two models agree on 96.6–97.2 % of the val segments. Two seeds with the same separator agree on 96.8–97.1 %. Changing the separator changes the predictions about as much as changing the seed.

## Conclusion
- **No difference: −0.003 ± 0.018.** One seed is better (+0.013), one is worse (−0.022) and one is equal. The mean is well inside seed noise.
- **Per class, nothing outside noise either.**
  - `cs` +0.013 and `ar` +0.007 are within one std.
  - `cc` −0.036 has std 0.047 and 88 val segments; it can't be read either way. `[SEP]` predicts `cc` less often, more precisely but with lower recall.
- **With `[SEP]`, all targets are predicted slightly less often.** `other` → `cs`, `aa` and `cc` drop by 14–32 segments each, and `aa` recall drops 0.545 → 0.519. This is a small shift towards `other`, not a better separation between classes.
- **The likely reason:** the token boundary is new information, but the model already found the boundaries without it. With `"\n"`, a segment boundary is usually also a change of wording, often a backchannel (e.g. "yeah", "uh-huh") or a new sentence start. And `[SEP]` says nothing about who is speaking, which is probably what matters for `aa` / `bk` / `cc` (who replies to whom). This is a hypothesis; this experiment doesn't test it.
- **Cost:** the same time per run (+7.5 % tokens didn't show up in the run time).
- **Status:** the separator doesn't matter at k = 3 at this setting. Which separator the later experiments use is open.
