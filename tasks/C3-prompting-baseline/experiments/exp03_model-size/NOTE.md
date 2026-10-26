## Question
Is exp01's letter bias a property of model size? Same setting as exp01 (zero-shot, k = 0, definitions on, v3 val) with larger Qwen2.5 Instruct models, raw and calibrated (as exp02).

## Change vs baseline
Only the model: `Qwen/Qwen2.5-1.5B-Instruct` (fp16), `Qwen/Qwen2.5-3B-Instruct` (fp16), `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4, fp16 compute: 7B in fp16 does not fit a 15 GB T4). Batch 16. Each run also calibrated with `scripts/calibrate_prompting.py`.
- Notebook `notebooks/kaggle_c3_exp03.ipynb`: 1.5B and 3B in parallel (one per T4), then 7B.

## Runs
- `c3_exp03_1.5b`, `c3_exp03_1.5b_cc`
- `c3_exp03_3b`, `c3_exp03_3b_cc`
- `c3_exp03_7b`, `c3_exp03_7b_cc`

## Result
pending

## Conclusion
pending
