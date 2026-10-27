## Question
Do solved examples in the prompt help 7B: does showing one example per label teach it what the annotators mean, especially for the short replies it confuses?

## Change vs baseline
Baseline: `c3_exp03_7b_v4` (Qwen2.5-7B-Instruct 4-bit, zero-shot, k = 0, definitions on, `template_v1.txt`; val macro-F1 0.207). Data v4. Notebook `notebooks/kaggle_c3_exp05-07.ipynb` (exp05, exp06 and exp07 in one session).
- Only change: `--shots-per-class 1 --shot-seed 0`: 7 train segments, one per label, in `LABELS` order (seg_ids in `settings.json`). The prompt grows from ~300 to ~440 tokens.

## Runs
- `c3_exp05_7b_shots1_v4`

## Result
pending

## Conclusion
pending
