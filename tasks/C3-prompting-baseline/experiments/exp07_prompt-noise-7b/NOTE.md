## Question
How much does the score move when only the wording of the instruction changes? Differences between experiments smaller than this spread are not real (task.md, "Prompt noise").

## Change vs baseline
Baseline: `c3_exp03_7b_v4` (Qwen2.5-7B-Instruct 4-bit, zero-shot, k = 0, definitions on, `template_v1.txt`; val macro-F1 0.207). Data v4. Notebook `notebooks/kaggle_c3_exp05-07.ipynb` (exp05, exp06 and exp07 in one session).
- Only change: the template, reworded twice with the same blanks, options and answer letters: `template_p1.txt` ("The following is an excerpt from a meeting transcript …"), `template_p2.txt` ("Read this extract from a recorded research meeting …").
- Run on the 7B at k = 0, the best setting so far (k = 1 is within 0.001).

## Runs
- `c3_exp07_7b_p1_v4`
- `c3_exp07_7b_p2_v4`

## Result
Commit `b9bb083`, notebook `notebooks/kaggle_c3_exp05-07.ipynb`, Kaggle T4, data v4 (val 16,188 segments). Speed: p1 4.0, p2 3.5 segments/s. Mean letter mass: p1 0.999, p2 0.970 (min 0.060).

| 7B, k = 0, v4 | macro-F1 | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|---|
| `template_v1` (baseline) | 0.207 | 0.379 | 0.138 | 0.041 | 0.219 | 0.050 | 0.414 | 0.742 |
| `template_p1` | 0.219 | 0.415 | 0.138 | 0.058 | 0.172 | 0.167 | 0.364 | 0.495 |
| `template_p2` | 0.242 | 0.360 | 0.141 | 0.050 | 0.203 | 0.524 | 0.176 | 0.572 |

**Spread of macro-F1 across the three wordings: 0.035** (0.207 – 0.242). Predicted `bk`: 5,768 / 10,459 / 8,365 (gold 1,029).

## Conclusion
Rewording only the instruction moves macro-F1 by up to 0.036, and individual classes far more (`ar` 0.050 / 0.167 / 0.524, `other` 0.742 / 0.495 / 0.572). This is the yardstick for the other experiments on this setting: differences below ~0.04 are not real.

Against it: few-shot (+0.091, exp05) is a real gain; definitions off (+0.010, exp06), context k = 1 (+0.001) and k = 3 (−0.022, exp04) are within noise. The per-class swings mean per-class conclusions from a single wording are unreliable.

Limits: three wordings, one model, zero-shot only; `template_p2` has a lower mean letter mass (0.970), so a few of its prompts were not answered with a letter.
