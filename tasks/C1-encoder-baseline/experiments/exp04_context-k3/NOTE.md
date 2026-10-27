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
Not written down before the runs. The plan made before exp02 was to run k = 3, 5 and 10 and pick the best k. The result is clear under any reasonable rule: every seed improves, by more than the seed noise.

## Result
All 3 runs finished, with no out-of-memory error. Each one's `settings.json` matches the setup (lr 1e-5, 3 epochs, `past_k = future_k = 3`, separator `"\n"`). 0 truncated train inputs. ~10.3 min per epoch, ~31 min per run (k = 0: ~5.5 and ~17), two runs in parallel.

Val macro-F1 (6), selected epoch in brackets. k = 0 is exp03's adopted cell, the same seeds.

| seed | k = 0 | k = 3 | difference |
|---|---|---|---|
| 42 | 0.5250 (3) | 0.5419 (3) | +0.017 |
| 43 | 0.5267 (3) | 0.5624 (3) | +0.036 |
| 44 | 0.5157 (3) | 0.5529 (2) | +0.037 |
| **mean ± std** | 0.5225 ± 0.0059 | **0.5524 ± 0.0103** | **+0.030 ± 0.011** |

Macro-F1 (7): 0.5807 ± 0.0052 → 0.6069 ± 0.0088.

Val macro-F1 (6) per epoch:

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| k = 0 | 0.450, 0.498, 0.525 | 0.366, 0.490, 0.527 | 0.422, 0.500, 0.516 |
| k = 3 | 0.425, 0.532, 0.542 | 0.447, 0.559, 0.562 | 0.449, 0.553, 0.549 |

Per-class F1, mean (std) over the 3 seeds:

| | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| k = 0 | 0.502 (0.013) | 0.429 (0.015) | 0.403 (0.006) | 0.612 (0.001) | 0.713 (0.008) | 0.476 (0.011) | 0.930 (0.001) |
| k = 3 | 0.525 (0.019) | 0.465 (0.015) | **0.547** (0.009) | 0.619 (0.006) | 0.705 (0.006) | 0.455 (0.027) | 0.934 (0.000) |
| difference | +0.023 | +0.036 | **+0.144** | +0.007 | −0.008 | −0.021 | +0.004 |

`aa` precision / recall: 0.469 / 0.353 → 0.550 / 0.545.

Confusion, mean over the 3 seeds, the rows that changed (gold → predicted):

| | k = 0 | k = 3 |
|---|---|---|
| `aa` → `aa` (of 817) | 288 | 445 |
| `aa` → `other` | 502 | 340 |
| `bk` → `aa` | 125 | 115 |
| `other` → `aa` | 200 | 247 |
| `cs` → `cs` (of 451) | 224 | 240 |
| `co` → `co` (of 111) | 41 | 46 |

## Conclusion
- **Context helps: +0.030** (0.5225 → 0.5524).
  - Every seed improves, by +0.017 to +0.037.
  - The worst k = 3 seed (0.5419) beats the best k = 0 seed (0.5267).
- **Most of the gain is `aa`: +0.144**, about 0.024 of the macro-F1's 0.030.
  - Its recall goes 0.35 → 0.55: about 160 fewer `aa` segments per seed are predicted as `other`.
  - `aa` vs `bk` barely changes (125 → 115 `bk` predicted as `aa`). The context mainly separates an acceptance from a segment that isn't a target at all, not from a backchannel.
  - The cost: ~47 more `other` segments per seed predicted as `aa`. Precision still rises, 0.47 → 0.55.
- **Smaller gains:** `co` +0.036, `cs` +0.023.
- **No gain, within noise:** `bk`, `ar`, `other`. `cc` is −0.021 with std 0.027; with 88 val segments it can't be read either way.
- **Spread:** 0.010 vs 0.006 at k = 0. Still well below the gain.
- **The curves flatten sooner.** At k = 3, epoch 2 → 3 adds only −0.004 to +0.010, against +0.016 to +0.037 at k = 0. Seed 44 is the first adopted-cell run that selected epoch 2. That's a hint for the later "do the k = 0 hyperparameters transfer to k > 0" experiment, not evidence.
- **Cost:** ~1.9× the time per epoch (10.3 vs 5.5 min) for 3.9× the tokens.
- **Next:** k = 5 and k = 10, the same setting and seeds. The k = 3 result is the reference to beat.
