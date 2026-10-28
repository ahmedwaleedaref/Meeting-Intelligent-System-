## Question
Do solved examples in the prompt help 7B: does showing one example per label teach it what the annotators mean, especially for the short replies it confuses?

## Change vs baseline
Baseline: `c3_exp03_7b_v4` (Qwen2.5-7B-Instruct 4-bit, zero-shot, k = 0, definitions on, `template_v1.txt`; val macro-F1 0.207). Data v4. Notebook `notebooks/kaggle_c3_exp05-07.ipynb` (exp05, exp06 and exp07 in one session).
- Only change: `--shots-per-class 1 --shot-seed 0`: 7 train segments, one per label, in `LABELS` order (seg_ids in `settings.json`). The prompt grows from ~300 to ~440 tokens.

## Runs
- `c3_exp05_7b_shots1_v4`

## Result
Commit `b9bb083`, notebook `notebooks/kaggle_c3_exp05-07.ipynb`, Kaggle T4, data v4 (val 16,188 segments). Speed 2.3 segments/s (~2 h). Mean letter mass 0.999 (min 0.090).

**val macro-F1 (6 targets) = 0.298** (baseline 0.207, +0.091).

| 7B, k = 0, v4 | macro-F1 | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|---|
| zero-shot (baseline) | 0.207 | 0.379 | 0.138 | 0.041 | 0.219 | 0.050 | 0.414 | 0.742 |
| 1 example per label | 0.298 | 0.362 | 0.284 | 0.290 | 0.071 | 0.429 | 0.350 | 0.831 |

Predicted: `other` 9,010 → 11,080 (gold 13,543), `bk` 5,768 → 818 (gold 1,029), `aa` 162 → 3,159 (gold 817), `ar` 50 → 364 (gold 149).
Gold `bk` predicted as: `aa` 567, `bk` 66, `other` 371. Gold `aa` found: 20 → 576 of 817.

## Conclusion
Few-shot is the largest gain so far (+0.091), well beyond the prompt-noise spread of 0.036 (exp07). One example per label moved the model towards the real label distribution: it predicts `other` for 11,080 segments instead of 9,010 (`other` F1 0.742 → 0.831), and `aa`, `ar`, `co` all improve (F1 0.041 → 0.290, 0.050 → 0.429, 0.138 → 0.284).

The short-reply confusion moved rather than disappeared: the model now sends acknowledgements to `aa` (567 of 1,029 gold `bk`), so `bk` drops to F1 0.071.

Limits: one set of examples (seed 0); a different draw could move the score, and that spread was not measured. The prompt-noise spread was measured on zero-shot prompts only.
