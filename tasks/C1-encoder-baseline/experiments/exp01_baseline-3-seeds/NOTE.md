# exp01: flat baseline, 3 seeds

## Question
What val macro-F1 (6 target classes) does the flat DeBERTa-v3-base classifier reach with the current default setup, and how much does it vary across seeds? This is the reference every later C1 experiment is compared against.

## Setup (change vs the earlier trial)
- **Same as the trial run of seed 42** (REPORT.md §5.2: fp32, val macro-F1 0.503): DeBERTa-v3-base, k = 0, batch 32, lr 2e-5, 5 epochs, warmup 10 %, weight decay 0.01, clipping 1.0, plain cross-entropy, best epoch by val macro-F1.
- **Changed:** fp16 mixed precision and fused AdamW (the `trial_run.py` defaults). Measured as speed-only changes; the default fp32 path still reproduces the earlier predictions byte for byte.
- **Seeds:** 42, 43, 44, as independent runs, two in parallel (one per T4).

## How it was run
- Notebook: `notebooks/kaggle_flat.ipynb` (Kaggle, GPU T4 × 2), branch `C1/encoder-baseline`, commit: _fill in from cell 2's `git log` line_.
- Command per seed (cell 5): `CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src python scripts/trial_run.py --mode flat --seeds <seed> --data-root <DATA_ROOT> --model microsoft/deberta-v3-base --out-dir /kaggle/working/runs`
- Run folders: `trial_flat_k0_seed42`, `trial_flat_k0_seed43`, `trial_flat_k0_seed44`.

## Result

| seed | run mode | selected epoch | val macro-F1 (6) | macro-F1 (7) | min per epoch (incl. val) | min per 50 steps |
|---|---|---|---|---|---|---|
| 42 | parallel (GPU 0) | 5 | 0.4985 | 0.5600 | 5.3 | 0.101 |
| 43 | parallel (GPU 1) | 3 | 0.5032 | 0.5641 | 5.3 | 0.102 |
| 44 | alone (GPU 0) | 2 | 0.4949 | 0.5574 | 5.4 | 0.103 |
| **mean ± std** | | | **0.4989 ± 0.0042** | **0.5605 ± 0.0034** | | |

Val macro-F1 (6) per epoch:
- seed 42: 0.411, 0.479, 0.472, 0.490, 0.499
- seed 43: 0.357, 0.480, 0.503, 0.494, 0.499
- seed 44: 0.438, 0.495, 0.484, 0.482, 0.488

Per-class F1, mean (std) over the 3 seeds:

| cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|
| 0.483 (0.017) | 0.383 (0.006) | 0.365 (0.049) | 0.606 (0.002) | 0.714 (0.010) | 0.442 (0.042) | 0.930 (0.002) |

## Conclusion
- **The flat baseline is val macro-F1 (6) = 0.499 ± 0.004** over 3 seeds. This is the reference for every later C1 experiment. A change needs a difference clearly larger than ~0.01 (the seed-to-seed range is 0.495–0.503) to count as an effect.
- **fp16 + fused AdamW didn't change the accuracy.** Seed 42: 0.4985 now vs 0.5031 in the fp32 trial, a difference the size of the seed-to-seed spread.
- **Speed: ~5.3 min per epoch including validation, ~27 min per seed,** against ~60 min for the fp32 trial (~2.2×).
- **Two runs in parallel don't slow each other down.** Seeds 42 and 43 in parallel took 0.101–0.102 min per 50 steps; seed 44 alone took 0.103. The shared CPU keeps up, so the second T4 doubles experiment throughput.
- **The val score plateaus after epoch 2–3 at about 0.48–0.50,** and the best epoch varies by seed (5, 3, 2). One seed is not enough to judge the number of epochs or the LR (task item 5).
- **Per class:** `aa` (accept) and `cc` (commitment) are the least stable across seeds (std 0.05 and 0.04); `bk`, `other` and `co` are very stable.
