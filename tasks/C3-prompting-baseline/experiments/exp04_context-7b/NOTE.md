## Question
Does a context window fix 7B's main error in exp03 (short replies such as "yeah" / "OK" predicted `bk` whatever they answer)?

## Change vs baseline
Baseline: exp03's `c3_exp03_7b` (Qwen2.5-7B-Instruct 4-bit, zero-shot, definitions on, k = 0; macro-F1 0.206). Only the context changes: `past_k = future_k = 1`, and `past_k = future_k = 3`.
- Data v4. Notebook `notebooks/kaggle_c3_v4.ipynb` (one session with the v4 reruns of exp01–exp03); the baseline is its `c3_exp03_7b_v4`.
- One factor at a time instead of the 16-cell grid: a 7B cell takes 1.5 h or more, so the full grid does not fit the GPU budget.

## Runs
- `c3_exp04_7b_k1_v4`
- `c3_exp04_7b_k3_v4`
- preliminary, on v3: `c3_exp04_7b_k1`, `c3_exp04_7b_k3` (commit `c0c893b`)

## Result
### v4 (reported)
commit `ec7f1d4` (same `src/` and prompt files as `c0c893b`), notebook `notebooks/kaggle_c3_v4.ipynb`, Kaggle T4, data v4 (val 16,188 segments, the encoder's data). Speed: k = 1 3.4, k = 3 2.8 segments/s.

| 7B | macro-F1 | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|---|
| k = 0 (exp03) | 0.207 | 0.379 | 0.138 | 0.041 | 0.219 | 0.050 | 0.414 | 0.742 |
| k = 1 | 0.208 | 0.277 | 0.123 | 0.088 | 0.166 | 0.275 | 0.320 | 0.385 |
| k = 3 | 0.185 | 0.240 | 0.113 | 0.180 | 0.170 | 0.198 | 0.209 | 0.296 |

Predicted `bk`: 5,768 (k = 0) → 10,112 (k = 1) → 9,389 (k = 3), gold 1,029; predicted `other`: 9,010 → 3,474 → 2,449, gold 13,543. `aa` found: 20 → 61 → 195 of 817.

### v3 (first runs, superseded; kept for the record)
Same code and prompt. Speed: k = 1 3.4 segments/s, k = 3 2.8 segments/s.

| 7B | macro-F1 | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|---|
| k = 0 (exp03) | 0.206 | 0.378 | 0.138 | 0.041 | 0.218 | 0.050 | 0.414 | 0.742 |
| k = 1 | 0.207 | 0.278 | 0.122 | 0.089 | 0.165 | 0.269 | 0.320 | 0.385 |
| k = 3 | 0.183 | 0.240 | 0.113 | 0.180 | 0.170 | 0.186 | 0.212 | 0.298 |

Predicted `bk`: 5,807 (k = 0) → 10,169 (k = 1) → 9,439 (k = 3); predicted `other`: 9,067 → 3,499 → 2,478 (gold `other`: 13,634). `aa` recall: 20 → 62 → 196 of 824.

## Conclusion
Context does not help overall: k = 1 equals k = 0 (0.208 vs 0.207), and k = 3 is lower (0.185). It does help the reply classes it was meant for (`aa` F1 0.041 → 0.180 at k = 3, `ar` 0.050 → 0.275 at k = 1), but with context the model reads most segments as replies: it predicts `bk` for 10,112 segments at k = 1 (gold 1,029), and `other` recall falls from 0.62 to 0.24. The precision of every target class drops, which cancels the gain.

The prompted model has no notion of how rare the target classes are (84 % `other`); with context it sees replies everywhere. The 7 scores allow shifting the `other`-vs-target balance without changing the prompt (task.md point 2), which is the most promising next lever.

Next: few-shot and definitions off at k = 0 (exp05, exp06): k = 0 scores as well as k = 1 and gives the shortest, fastest prompts.
