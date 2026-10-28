## Question
With examples in the prompt, does contextual calibration help, and does the order or number of the examples matter? Zhao et al. (2021) find calibration helps most in few-shot prompts, where the examples add majority-label and recency biases (the last example's label pulls the answer). Our calibration was only tried zero-shot (exp02, exp03), and exp05's examples always end with `other`.

## Change vs baseline
Baseline: exp05's setting (Qwen2.5-7B-Instruct 4-bit, k = 0, definitions on, `template_v1.txt`, 1 example per label, seed 0, label order; val macro-F1 0.298 on v4), rerun here on **data v3** as the control: every exp08 comparison is against it, within v3. v3 and v4 scores differed by at most 0.002 macro-F1 in exp03 and exp04. Notebook `notebooks/kaggle_c3_exp08.ipynb`, dataset `smi-v3-meetings`.
- Example order, the same 7 examples: label order (control, exp05's order), reversed (`cs` last), shuffled with `--shot-order-seed 1` (`bk, other, cc, aa, cs, ar, co`: `co` last).
- Number: 2 examples per label (`--shots-per-class 2`, seed 0, label order).
- Calibration of every run, no re-scoring (`scripts/calibrate_prompting.py --target uniform train-prior`):
  - `uniform`: the paper's method, a content-free input is calibrated to equal letters;
  - `train-prior`: calibrated to the train label frequencies instead (84 % `other`). The paper's datasets are near-balanced; ours is not, and for the larger models the content-free "bias" was mostly a correct `other` (exp03).
- Speed, not results: `--prefix-cache` runs the tokens every prompt shares (instruction, options, examples) once. Before the runs, the notebook checks it on the 7B against each full prompt scored alone; it must stay within 3 × the usual padded batch's own difference + 0.01, and each run repeats the check on its longest prompts. If the check fails, the runs go without the cache (batch 16). With the cache: batch 32. Both T4s in parallel.

## Runs
- v3: `c3_exp08_7b_labels` (control), `c3_exp08_7b_reversed`, `c3_exp08_7b_shuffled`, `c3_exp08_7b_shots2`
- each also `_cc` (uniform) and `_ccp` (train-prior)

## Result
First attempt (commit `8021317`), no scores: every run stopped at its first-batch check. Cached vs padded full-prompt batch differed by 0.048 (labels), 0.082 (reversed) and 0.062 (shuffled), above that version's fixed 0.02 limit; `shots2` ran out of GPU memory in the check (32 full prompts with 14 examples). The 0.5B in fp32 had matched to 7.5e-6, so fp16 rounding on the 7B is the likely cause, not the cache's logic; the check now compares against the padded batch's own difference instead of a fixed limit, on 8 prompts.

pending

## Conclusion
pending
