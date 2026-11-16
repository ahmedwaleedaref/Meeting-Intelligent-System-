## Question
Does contextual calibration (Zhao et al. 2021) remove exp01's letter bias and recover a usable classifier, without re-scoring the split?

## Change vs baseline
exp01's saved scores (`scores.jsonl` of `c3_exp01_v4`; first `c3_exp01_zs_k0_defs` on v3), each divided by the model's letter probabilities on content-free inputs ("N/A", "", "[MASK]") in the same prompt, then renormalised. Script `scripts/calibrate_prompting.py`; no new GPU run.
- The bias was measured on a laptop CPU in fp32, while exp01's scores came from a T4 in fp16; the difference is negligible next to the size of the bias.

## Runs
- v4: `c3_exp01_v4_cc` (from `c3_exp01_v4`)
- v3: `c3_exp01_zs_k0_defs_cc` (from `c3_exp01_zs_k0_defs`)

## Result
### v4 (reported)
commit `ec7f1d4` (same `src/` and prompt files as `c0c893b`), notebook `notebooks/kaggle_c3_v4.ipynb`, Kaggle T4, data v4 (val 16,188 segments, the encoder's data). **val macro-F1 = 0.012** (raw 0.019).

| | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| F1 | 0.017 | 0.024 | 0.031 | 0.002 | 0.000 | 0.000 | 0.359 |

### v3 (first runs, superseded; kept for the record)
Measured bias (content-free probabilities): cs 0.214, co 0.124, aa 0.040, bk 0.045, ar 0.130, cc 0.370, other 0.077.

**val macro-F1 (6 targets) = 0.012** (exp01: 0.019).

| | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| F1 | 0.017 | 0.025 | 0.030 | 0.002 | 0.000 | 0.000 | 0.367 |
| predicted | 2,662 | 46 | 8,242 | 200 | 1 | 2 | 5,136 |

## Conclusion
Calibration moved the predictions, but did not make them correct: `other` rose from 235 to 5,136 predictions (F1 0.023 → 0.367), and most of the rest now goes to `aa` (8,242), whose content-free probability was among the lowest. The target classes did not improve. Calibration can only remove a constant preference; it cannot create information the scores don't contain. At 0.5B zero-shot, the scores carry almost no class information, so model size is the next question (exp03).
