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
Commit `c0c893b`, data v3 (val, 16,289 segments), the same prompt files as exp01 (sha256 template `80608d05…`, definitions `945cca1c…`). Kaggle 2 × T4: 1.5B and 3B in parallel (17.6 and 10.5 segments/s, ~15 and ~26 min), then 7B (3.3 segments/s, ~82 min). No run failed.

**val macro-F1 (6 targets): 0.5B 0.019 → 1.5B 0.045 → 3B 0.107 → 7B 0.206.** The answer format is followed at every size (mean letter mass ≥ 0.996).

| | macro-F1 | calibrated | `other` F1 | letter mass (min) | most predicted |
|---|---|---|---|---|---|
| 0.5B (exp01, exp02) | 0.019 | 0.012 | 0.023 | 0.996 (0.965) | A `cs` 72.8 % |
| 1.5B | 0.045 | 0.057 | 0.356 | 0.997 (0.089) | D `bk` 48.7 % |
| 3B | 0.107 | 0.089 | 0.423 | 0.996 (0.348) | E `ar` 40.7 % |
| 7B (4-bit) | 0.206 | 0.182 | 0.742 | 1.000 (0.810) | G `other` 55.7 % |

Per-class F1:

| | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| 1.5B | 0.020 | 0.000 | 0.046 | 0.207 | 0.000 | 0.000 | 0.356 |
| 1.5B cal. | 0.042 | 0.000 | 0.078 | 0.222 | 0.000 | 0.000 | 0.098 |
| 3B | 0.265 | 0.018 | 0.002 | 0.305 | 0.014 | 0.036 | 0.423 |
| 3B cal. | 0.190 | 0.018 | 0.014 | 0.247 | 0.043 | 0.020 | 0.138 |
| 7B | 0.378 | 0.138 | 0.041 | 0.218 | 0.050 | 0.414 | 0.742 |
| 7B cal. | 0.183 | 0.032 | 0.061 | 0.255 | 0.363 | 0.195 | 0.406 |

Predicted labels (raw): 1.5B `bk` 7,933, `aa` 5,052, `other` 3,254; 3B `ar` 6,631, `bk` 5,040, `other` 4,051; 7B `other` 9,067, `bk` 5,807, `cs` 1,044. A random guesser with the same label counts would score 0.034 (1.5B), 0.025 (3B) and 0.030 (7B).

Measured bias (content-free probabilities): 1.5B `other` 0.634, `bk` 0.297, `aa` 0.058; 3B `other` 0.544, `ar` 0.250, `cc` 0.157; 7B `other` 0.725, `bk` 0.275, every other letter < 0.001.

## Conclusion
Size is the main lever so far: macro-F1 grows tenfold from 0.5B to 7B, and only the 3B and 7B are clearly above chance. The 0.5B's first-letter preference is gone, but each size has its own dominant answer: 1.5B `bk`/`aa`, 3B `ar` (6,056 of its 6,631 `ar` predictions are `other`), and only the 7B predicts `other` most often.

The 7B recognises the classes with distinctive wording (`cc` 0.414, `cs` 0.378), not the short replies: 631 of 824 `aa` and 4,232 of 13,634 `other` segments are predicted `bk`. With the target alone (k = 0), "yeah" or "okay" reads the same whether it accepts, acknowledges or is a backchannel, the pair the MRDA manual names as most confused. Context (k > 0) in the grid is the lever for this.

Calibration helps only the 1.5B (0.045 → 0.057) and lowers the 3B and 7B. For the larger models, the content-free "bias" is mostly `other` and `bk`: an empty segment arguably is `other` or a backchannel, so that preference is partly a correct answer, and dividing it out pushes predictions away from `other` (7B: 9,067 → 3,723 `other` predictions, `other` F1 0.742 → 0.406). Raw scores are kept for the larger models.

Limits: the 7B is 4-bit and the others fp16, so size and quantization are not fully separated. Cost: one 7B val run at k = 0, zero-shot takes ~82 min on a T4; context and examples lengthen every prompt, so a 16-cell grid on the 7B would take well over 22 GPU-hours, against ~26 min per run for the 3B.
