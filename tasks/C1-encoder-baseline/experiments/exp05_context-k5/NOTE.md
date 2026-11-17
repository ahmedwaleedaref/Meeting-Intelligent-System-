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
- **First attempt, plain batch 32: out of memory.**
  - Seeds 42 and 44 (GPU 0) crashed in epoch 1. At the crash: 13.4 GiB allocated + ~1.0 GiB reserved but unallocated, of the T4's 14.56 GiB. The estimate was ~14.7 GB (13.7 GiB).
  - The ~1 GiB not usable is fragmentation from batches of changing length. A memory check that repeats one batch shape doesn't build it up.
  - The interrupted runs were stopped and deleted.
  - Memory check result: _what it printed_.
- **Rerun with gradient checkpointing**, by the rule above. Memory check: _peak allocated / reserved, from section 5_.
- **Command per seed:** `CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src python scripts/trial_run.py --mode flat --seeds <seed> --lr 1e-5 --epochs 3 --past-k 5 --future-k 5 --grad-checkpointing --name exp05_k5 --data-root <DATA_ROOT> --model microsoft/deberta-v3-base --out-dir /kaggle/working/runs`
- **Run folders:** `exp05_k5_seed42`, `exp05_k5_seed43`, `exp05_k5_seed44`.

## Decision rule
Not written down before the runs (the same as exp04). The plan made before exp02 was to run k = 3, 5 and 10 and pick the best k.

## Result
All 3 runs finished with gradient checkpointing and no out-of-memory error, with 0 truncated train inputs and 2466 batches per epoch. ~17.2 min per epoch on GPU 0 (seeds 42, 44) and ~18.6 on GPU 1 (seed 43): ~52–56 min per run, against ~31 at k = 3.

**The run folders were lost** (the downloaded `exp05_runs.zip` is gone). What's left is the printed output of the results cell: macro-F1 (6) and (7), per-class F1 at the selected epoch, and macro-F1 (6) per epoch. The confusion matrices, per-class F1 per epoch and prediction counts are lost.

Val macro-F1 (6), selected epoch in brackets:

| seed | k = 3 (exp04) | k = 5 | difference |
|---|---|---|---|
| 42 | 0.5419 (3) | 0.5529 (3) | +0.011 |
| 43 | 0.5624 (3) | 0.5477 (3) | −0.015 |
| 44 | 0.5529 (2) | 0.4659 (3) | −0.087 |
| **mean ± std** | 0.5524 ± 0.0103 | **0.5222 ± 0.0488** | −0.030 ± 0.051 |

Macro-F1 (7): 0.6069 ± 0.0088 → 0.5811 ± 0.0417.

Val macro-F1 (6) per epoch:

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| k = 3 | 0.425, 0.532, 0.542 | 0.447, 0.559, 0.562 | 0.449, 0.553, 0.549 |
| k = 5 | 0.401, 0.464, 0.553 | 0.404, 0.527, 0.548 | 0.397, 0.465, 0.466 |

Per-class F1, mean (std) over the 3 seeds:

| | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| k = 3 | 0.525 (0.019) | 0.465 (0.015) | 0.547 (0.009) | 0.619 (0.006) | 0.705 (0.006) | 0.455 (0.027) | 0.934 (0.000) |
| k = 5 | 0.541 (0.008) | 0.444 (0.034) | 0.527 (0.009) | 0.616 (0.001) | 0.705 (0.008) | **0.299 (0.259)** | 0.935 (0.001) |
| difference | +0.016 | −0.021 | −0.020 | −0.003 | 0.000 | −0.156 | +0.001 |

`cc` per seed at k = 5: 0.459 / 0.438 / **0.000**.

## Conclusion
- **k = 5 doesn't beat k = 3.** The mean drops by 0.030, but almost all of it is one class in one seed.
- **Seed 44 lost `cc` completely (F1 0.000).** Without it, seed 44's 5 other targets average 0.559, the same as seeds 42 and 43 (0.571, 0.570). With a typical `cc` of ~0.45 its macro-F1 (6) would be ~0.54.
  - Whether the model predicted no `cc` at all, or predicted it only wrongly, can't be checked: the confusion matrix is lost.
  - It's the rarest target (88 val segments), and a plain cross-entropy loss gives it no extra weight. This is the failure that the class-weighting experiment is meant to address.
- **Seeds 42 and 43, `cc` intact:** 0.5503 vs k = 3's 0.5522 on the same seeds. No difference.
- **Without `cc` at all** (the 5 other targets, mean over seeds): 0.567 vs 0.572 at k = 3. Also no gain.
- **Per class:** `cs` +0.016, `co` −0.021, `aa` −0.020. All within about two stds; nothing that more context clearly helps.
- **It learns more slowly.** At epoch 2, k = 5 is at 0.464 / 0.527 / 0.465, against 0.532 / 0.559 / 0.553 at k = 3. Seed 42 gains +0.089 in epoch 3, so it may not be done. The lr 1e-5, 3 epochs setting was tuned at k = 0. This is more evidence for the "do the hyperparameters transfer to k > 0" experiment, not proof that k = 5 would win with more epochs.
- **Cost:** ~1.7× the time per run of k = 3 (~52–56 vs ~31 min), including ~1/3 from gradient checkpointing.
- **Status:** k = 3 stays the reference. Whether k = 10 still runs at this setting is open.
