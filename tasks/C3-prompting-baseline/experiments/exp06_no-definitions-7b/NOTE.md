## Question
Do the label definitions (from the MRDA manual) help 7B, or are the label names enough?

## Change vs baseline
Baseline: `c3_exp03_7b_v4` (Qwen2.5-7B-Instruct 4-bit, zero-shot, k = 0, definitions on, `template_v1.txt`; val macro-F1 0.207). Data v4. Notebook `notebooks/kaggle_c3_exp05-07.ipynb` (exp05, exp06 and exp07 in one session).
- Only change: `--options names_v1.txt` (`A) suggestion` … `G) other`, no definitions). The prompt shrinks from ~300 to ~110 tokens.

## Runs
- `c3_exp06_7b_names_v4`

## Result
Commit `b9bb083`, notebook `notebooks/kaggle_c3_exp05-07.ipynb`, Kaggle T4, data v4 (val 16,188 segments). Speed 10.1 segments/s (~27 min, 2.8 × faster than with definitions).

**val macro-F1 (6 targets) = 0.217** (with definitions 0.207).

| 7B, k = 0, v4 | macro-F1 | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|---|
| definitions on (baseline) | 0.207 | 0.379 | 0.138 | 0.041 | 0.219 | 0.050 | 0.414 | 0.742 |
| names only | 0.217 | 0.377 | 0.158 | 0.000 | 0.171 | 0.557 | 0.043 | 0.482 |

Predicted without definitions: `bk` 10,466, `aa` 8, `cc` 6, `other` 4,685.

## Conclusion
On the average, no difference: +0.010 is well inside the prompt-noise spread (0.036, exp07). Per class, the definitions matter a lot but in both directions: without them the model never answers `aa` (F1 0.000) and almost never `cc` (0.043 vs 0.414), while `ar` rises (0.557 vs 0.050) and `bk` is predicted for 10,466 segments. The definitions change *which* classes the model uses, not how well it does overall.

Names only is 2.8 × faster, so it is the cheaper choice when the average is what matters; the per-class picture says the wording of the options is itself a strong lever.
