# exp03: confirm the two best exp02 cells

## Question
exp02 found the two best cells with one seed each: (1e-5, 3 epochs) = 0.5250 and (5e-5, 3 epochs) = 0.5254. Do they hold with 3 seeds? Which one becomes the fixed setting for the context experiments, and the new C1 reference in place of exp01's 0.499 ± 0.004?

## Setup (change vs exp02)
- **Same as exp02:** DeBERTa-v3-base, k = 0, batch 32, warmup 10 %, weight decay 0.01, clipping 1.0, plain cross-entropy, fp16 + fused AdamW, best epoch by val macro-F1.
- **Cells:** (1e-5, 3 epochs) and (5e-5, 3 epochs).
- **Seeds:** 43 and 44. Seed 42 of each cell is exp02's run, so 4 new runs.

## How it was run
- **Notebook:** `notebooks/kaggle_exp03_confirm_top_cells.ipynb` (Kaggle, GPU T4 × 2, two runs at a time). Branch `C1/encoder-baseline`, commit: _fill in from the `git log` line of the "Code and packages" cell_.
- **Command per run:** `CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src python scripts/trial_run.py --mode flat --seeds <seed> --lr <lr> --epochs 3 --name exp03_lr<lr>_ep3 --data-root <DATA_ROOT> --model microsoft/deberta-v3-base --out-dir /kaggle/working/runs`
- **Run folders:** `exp03_lr1e-05_ep3_seed43`, `exp03_lr1e-05_ep3_seed44`, `exp03_lr5e-05_ep3_seed43`, `exp03_lr5e-05_ep3_seed44`.

## Decision rule
Not written down before the runs. The result is clear under any reasonable rule: the adopted cell beats the old setting and the other cell by ≥ 0.01, with the smallest spread.

## Result
All 4 runs finished. Each one's `settings.json` matches its cell. ~5.5 min per epoch, ~17 min per run.

Val macro-F1 (6), selected epoch in brackets. Seed 42 is from exp02.

| cell | seed 42 | seed 43 | seed 44 | mean ± std | macro-F1 (7) mean ± std |
|---|---|---|---|---|---|
| 1e-5, 3 epochs | 0.5250 (3) | 0.5267 (3) | 0.5157 (3) | **0.5225 ± 0.0059** | 0.5807 ± 0.0052 |
| 5e-5, 3 epochs | 0.5254 (3) | 0.4973 (3) | 0.5089 (3) | **0.5105 ± 0.0142** | 0.5705 ± 0.0121 |
| exp01: 2e-5, 5 epochs | 0.4985 (5) | 0.5032 (3) | 0.4949 (2) | 0.4989 ± 0.0042 | 0.5605 ± 0.0034 |

Val macro-F1 (6) per epoch:

| cell | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| 1e-5, 3 | 0.450, 0.498, 0.525 | 0.366, 0.490, 0.527 | 0.422, 0.500, 0.516 |
| 5e-5, 3 | 0.362, 0.485, 0.525 | 0.372, 0.473, 0.497 | 0.439, 0.466, 0.509 |

Per-class F1, mean (std) over the 3 seeds:

| cell | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| 1e-5, 3 | 0.502 (0.013) | 0.429 (0.015) | 0.403 (0.006) | 0.612 (0.001) | 0.713 (0.008) | 0.476 (0.011) | 0.930 (0.001) |
| 5e-5, 3 | 0.493 (0.000) | 0.431 (0.038) | 0.400 (0.006) | 0.612 (0.006) | 0.710 (0.001) | 0.418 (0.041) | 0.930 (0.001) |
| exp01 | 0.483 (0.017) | 0.383 (0.006) | 0.365 (0.049) | 0.606 (0.002) | 0.714 (0.010) | 0.442 (0.042) | 0.930 (0.002) |

## Conclusion
- **Adopted: lr 1e-5, 3 epochs.** It's the fixed setting for the context experiments, and **0.5225 ± 0.0059** is the new k = 0 reference, replacing exp01's 0.499 ± 0.004.
- **The gain over exp01 is real: +0.024.**
  - The worst seed (0.5157) beats exp01's best (0.5032).
  - Seed by seed it's +0.021 to +0.027.
  - It's broad: `co` +0.046, `aa` +0.038, `cc` +0.034, `cs` +0.019; `bk`, `ar` and `other` are unchanged.
  - `aa` became much more stable (std 0.049 → 0.006).
- **(5e-5, 3 epochs) didn't hold up: 0.5105 ± 0.0142.** Its exp02 seed (0.5254) was its lucky draw, and its other two seeds scored 0.497 and 0.509. With the short schedule, the high LR is unstable, mostly through `cc` and `co` (std ~0.04). That's why a single-seed grid only screens.
- **Cheaper too:** ~17 min per run instead of ~28.
- **Left for later, as its own experiment:** whether the k = 0 setting (LR, and 3 epochs possibly being the edge of the grid) is also the best for k > 0.
