# exp02: LR × epochs grid, k = 0

## Question
Which learning rate and number of epochs give the best val macro-F1 (6 target classes) for the flat DeBERTa-v3-base classifier at k = 0? Is the exp01 setting (2e-5, 5 epochs) handicapping the baseline? The chosen setting is then fixed for the context experiments (k > 0).

## Setup (change vs exp01)
- **Same as exp01:** DeBERTa-v3-base, k = 0, batch 32, warmup 10 %, weight decay 0.01, clipping 1.0, plain cross-entropy, fp16 + fused AdamW, best epoch by val macro-F1.
- **Changed:**
  - LR ∈ {1e-5, 2e-5, 3e-5, 5e-5};
  - epochs ∈ {3, 5, 8}.
- **Seed:** 42 for every cell.
- **The number of epochs is the length of the LR schedule.** The LR decays linearly to 0 at the last epoch, so "3 epochs" is a different schedule from "epoch 3 of a 5-epoch run". Within each run, the best epoch is still selected on val.
- **The cell (2e-5, 5)** is exp01's seed-42 run (`trial_flat_k0_seed42`): the same code, seed and settings. So 11 new runs.

## How it was run
- **Notebook:** `notebooks/kaggle_exp02_lr_epochs.ipynb` (Kaggle, GPU T4 × 2, two runs at a time). Branch `C1/encoder-baseline`, commit: _fill in from the `git log` line of the "Code and packages" cell_.
- **Command per cell:** `CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src python scripts/trial_run.py --mode flat --seeds 42 --lr <lr> --epochs <epochs> --name exp02_lr<lr>_ep<epochs> --data-root <DATA_ROOT> --model microsoft/deberta-v3-base --out-dir /kaggle/working/runs`
- **Run folders:** `exp02_lr<lr>_ep<epochs>_seed42`, e.g. `exp02_lr3e-05_ep8_seed42`.

## Decision rule
Not written down before the results were read. The rule decided afterwards:
- a single-seed grid only screens;
- the two best cells are confirmed with 2 more seeds each in [exp03](../exp03_confirm-top-cells/NOTE.md);
- 2e-5 gets no advantage for being the common default.

## Result
All 11 runs finished. Each one's `settings.json` matches its cell. About 5.6 min per epoch including validation, two runs in parallel.

Val macro-F1 (6), seed 42, selected epoch in brackets. The (2e-5, 5) cell is exp01's seed 42.

| LR \ epochs | 3 | 5 | 8 | mean over epochs |
|---|---|---|---|---|
| 1e-5 | **0.5250** (3) | 0.5133 (4) | 0.4964 (7) | 0.5116 |
| 2e-5 | **0.5199** (3) | 0.4985 (5) | 0.4929 (5) | 0.5038 |
| 3e-5 | **0.5166** (3) | 0.5033 (5) | 0.4882 (8) | 0.5027 |
| 5e-5 | **0.5254** (3) | 0.4977 (5) | 0.4996 (7) | 0.5076 |
| **mean over LRs** | **0.5217** | 0.5032 | 0.4943 | |

Val macro-F1 (6) per epoch:

| cell | curve |
|---|---|
| 1e-5, 3 | 0.450, 0.498, 0.525 |
| 2e-5, 3 | 0.432, 0.496, 0.520 |
| 3e-5, 3 | 0.422, 0.479, 0.517 |
| 5e-5, 3 | 0.362, 0.485, 0.525 |
| 1e-5, 5 | 0.379, 0.484, 0.504, 0.513, 0.512 |
| 2e-5, 5 | 0.411, 0.479, 0.472, 0.490, 0.499 |
| 3e-5, 5 | 0.434, 0.483, 0.449, 0.499, 0.503 |
| 5e-5, 5 | 0.389, 0.432, 0.466, 0.491, 0.498 |
| 1e-5, 8 | 0.363, 0.454, 0.452, 0.485, 0.493, 0.491, 0.496, 0.493 |
| 2e-5, 8 | 0.365, 0.483, 0.436, 0.467, 0.493, 0.481, 0.491, 0.471 |
| 3e-5, 8 | 0.397, 0.461, 0.459, 0.464, 0.476, 0.486, 0.478, 0.488 |
| 5e-5, 8 | 0.336, 0.443, 0.438, 0.428, 0.436, 0.494, 0.500, 0.475 |

Training loss at the last epoch:
- 3 epochs: 0.30–0.32;
- 5 epochs: 0.25–0.29;
- 8 epochs: 0.22–0.25.

Per-class F1:

| cell | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| 1e-5, 3 | 0.514 | 0.419 | 0.407 | 0.611 | 0.711 | 0.488 | 0.931 |
| 2e-5, 3 | 0.503 | 0.436 | 0.400 | 0.611 | 0.709 | 0.461 | 0.929 |
| 3e-5, 3 | 0.480 | 0.442 | 0.402 | 0.611 | 0.718 | 0.446 | 0.929 |
| 5e-5, 3 | 0.493 | 0.474 | 0.404 | 0.611 | 0.711 | 0.461 | 0.930 |
| 1e-5, 5 | 0.469 | 0.429 | 0.403 | 0.614 | 0.711 | 0.454 | 0.931 |
| 2e-5, 5 | 0.476 | 0.376 | 0.400 | 0.605 | 0.711 | 0.424 | 0.928 |
| 3e-5, 5 | 0.472 | 0.396 | 0.393 | 0.606 | 0.704 | 0.449 | 0.928 |
| 5e-5, 5 | 0.438 | 0.388 | 0.397 | 0.611 | 0.707 | 0.444 | 0.928 |
| 1e-5, 8 | 0.475 | 0.404 | 0.398 | 0.612 | 0.690 | 0.400 | 0.927 |
| 2e-5, 8 | 0.437 | 0.386 | 0.399 | 0.613 | 0.683 | 0.439 | 0.929 |
| 3e-5, 8 | 0.460 | 0.400 | 0.369 | 0.610 | 0.706 | 0.383 | 0.929 |
| 5e-5, 8 | 0.452 | 0.402 | 0.393 | 0.604 | 0.697 | 0.448 | 0.927 |

## Conclusion
- **3 epochs beat 5 and 8 at every LR.**
  - All four 3-epoch cells (0.517–0.525) are above every 5- and 8-epoch cell, and above exp01's 3-seed mean (0.499 ± 0.004) by ~0.02.
  - The same direction at 4 LRs is stronger evidence than any single cell.
  - It also makes runs ~40 % cheaper (~17 min instead of ~28).
- **Longer schedules overfit.** Training loss keeps falling with more epochs (0.30 → 0.22), while val macro-F1 drops. The rare classes gain most from the short schedule: `cc` 0.45–0.49 at 3 epochs vs 0.38–0.45 at 8.
- **It's the full LR decay that helps, not stopping early.**
  - Every 3-epoch run selected its last epoch, where the LR had just reached 0.
  - Epoch 3 inside a 5-epoch run, with the LR still high, scored only 0.45–0.50.
- **The grid can't separate the LRs.** At 3 epochs they span 0.517–0.525, not in monotonic order: smaller than one seed's noise.
- **Next:**
  - confirm the two best cells, (1e-5, 3) and (5e-5, 3), with seeds 43 and 44: exp03;
  - open question: the best may lie below the grid's edge. 3 is the smallest value tested, and its curves were still rising.
