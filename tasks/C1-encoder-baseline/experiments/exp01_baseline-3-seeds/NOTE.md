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
_To fill in after the run:_
- per seed: selected epoch and val macro-F1 (6), and per-class F1;
- mean and standard deviation over the 3 seeds;
- time per epoch and per seed (and whether two parallel runs slowed each other down, from the step lines);
- seed 42 vs the earlier fp32 trial (0.503): fp16 + fused should change speed, not accuracy.

## Conclusion
_To fill in._
