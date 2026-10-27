## Question
Do the label definitions (from the MRDA manual) help 7B, or are the label names enough?

## Change vs baseline
Baseline: `c3_exp03_7b_v4` (Qwen2.5-7B-Instruct 4-bit, zero-shot, k = 0, definitions on, `template_v1.txt`; val macro-F1 0.207). Data v4. Notebook `notebooks/kaggle_c3_exp05-07.ipynb` (exp05, exp06 and exp07 in one session).
- Only change: `--options names_v1.txt` (`A) suggestion` … `G) other`, no definitions). The prompt shrinks from ~300 to ~110 tokens.

## Runs
- `c3_exp06_7b_names_v4`

## Result
pending

## Conclusion
pending
