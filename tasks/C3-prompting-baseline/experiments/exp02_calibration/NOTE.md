## Question
Does contextual calibration (Zhao et al. 2021) remove exp01's letter bias and recover a usable classifier, without re-scoring the split?

## Change vs baseline
exp01's saved scores (`scores.jsonl` of `c3_exp01_zs_k0_defs`), each divided by the model's letter probabilities on content-free inputs ("N/A", "", "[MASK]") in the same prompt, then renormalised. Script `scripts/calibrate_prompting.py`; no new GPU run.
- The bias was measured on a laptop CPU in fp32, while exp01's scores came from a T4 in fp16; the difference is negligible next to the size of the bias.

## Runs
- `c3_exp01_zs_k0_defs_cc` (from `c3_exp01_zs_k0_defs`)

## Result
Measured bias (content-free probabilities): cs 0.214, co 0.124, aa 0.040, bk 0.045, ar 0.130, cc 0.370, other 0.077.

**val macro-F1 (6 targets) = 0.012** (exp01: 0.019).

| | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| F1 | 0.017 | 0.025 | 0.030 | 0.002 | 0.000 | 0.000 | 0.367 |
| predicted | 2,662 | 46 | 8,242 | 200 | 1 | 2 | 5,136 |

## Conclusion
Calibration moved the predictions, but did not make them correct: `other` rose from 235 to 5,136 predictions (F1 0.023 → 0.367), and most of the rest now goes to `aa` (8,242), whose content-free probability was among the lowest. The target classes did not improve. Calibration can only remove a constant preference; it cannot create information the scores don't contain. At 0.5B zero-shot, the scores carry almost no class information, so model size is the next question (exp03).
