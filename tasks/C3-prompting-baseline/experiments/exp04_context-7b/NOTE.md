## Question
Does a context window fix 7B's main error in exp03 (short replies such as "yeah" / "OK" predicted `bk` whatever they answer)?

## Change vs baseline
Baseline: exp03's `c3_exp03_7b` (Qwen2.5-7B-Instruct 4-bit, zero-shot, definitions on, k = 0; macro-F1 0.206). Only the context changes: `past_k = future_k = 1`, and `past_k = future_k = 3`.
- Data v4. Notebook `notebooks/kaggle_c3_v4.ipynb` (one session with the v4 reruns of exp01–exp03); the baseline is its `c3_exp03_7b_v4`.
- One factor at a time instead of the 16-cell grid: a 7B cell takes 1.5 h or more, so the full grid does not fit the GPU budget.

## Runs
- `c3_exp04_7b_k1_v4`
- `c3_exp04_7b_k3_v4`
- (a first launch on v3, `c3_exp04_7b_k1` / `c3_exp04_7b_k3`, was superseded by the v4 data before it was analysed)

## Result
pending

## Conclusion
pending
