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
_To fill in before reading the results:_
- the size of a difference that counts (exp01's seed spread: 0.495–0.503);
- whether the best cell gets 2 more seeds before it's adopted;
- what wins a tie.

## Result
_Pending._

## Conclusion
_Pending._
