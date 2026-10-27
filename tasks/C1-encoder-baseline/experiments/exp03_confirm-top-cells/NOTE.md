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
_Fill in before the runs:_
- **Is the 3-epoch gain over exp01 real?** For example: the cell's 3-seed mean beats 0.499 by more than ~0.01.
- **Which cell is adopted** if both pass, and what wins a tie between them (their means within ~0.01)?
- **What happens if neither holds up?**

## Result
_Pending._

## Conclusion
_Pending._
