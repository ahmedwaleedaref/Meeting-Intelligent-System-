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

### Second attempt (reported)
Commit `d78b088`, notebook `notebooks/kaggle_c3_exp08.ipynb`, Kaggle 2 × T4, data v3 (val 16,289 segments). The cache passed its check and was used: 442 shared tokens (629 with 2 examples per label). Largest difference from each full prompt scored alone: cached 0.047 / 0.087 / 0.112 / 0.050, padded batch 0.051 / 0.119 / 0.112 / 0.050 (labels / reversed / shuffled / shots2): the cache is as exact as the usual batching. Speed 18–25 segments/s (exp05: 2.3), 11–15 min per run, 34 min for the whole notebook. Mean letter mass ≥ 0.999.

| 7B, k = 0, v3 | last example | macro-F1 | calibrated, uniform | calibrated, train frequencies |
|---|---|---|---|---|
| 1 per label, label order (control) | `other` | 0.271 | 0.174 | 0.253 |
| 1 per label, reversed | `cs` | 0.214 | 0.178 | 0.227 |
| 1 per label, shuffled (seed 1) | `co` | 0.278 | 0.260 | 0.298 |
| **2 per label, label order** | `other` | **0.341** | 0.284 | 0.335 |

Per-class F1 (raw):

| | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| control | 0.334 | 0.185 | 0.292 | 0.116 | 0.371 | 0.326 | 0.767 |
| reversed | 0.237 | 0.095 | 0.236 | 0.154 | 0.281 | 0.283 | 0.467 |
| shuffled | 0.333 | 0.181 | 0.265 | 0.150 | 0.408 | 0.331 | 0.645 |
| 2 per label | 0.372 | 0.215 | 0.376 | 0.139 | 0.548 | 0.397 | 0.882 |

Predicted (raw, control / reversed / shuffled / 2 per label): `other` 10,047 / 4,806 / 7,489 / 13,151 (gold 13,634); `cs` 744 / 2,327 / 813 / 558 (gold 454).

Measured bias (content-free probabilities): control `other` 0.964; reversed `other` 0.465, `cs` 0.392, `bk` 0.121; shuffled `cs` 0.357, `bk` 0.346, `cc` 0.106; 2 per label `other` 0.766, `aa` 0.091, `bk` 0.081. Train frequencies (the `_ccp` target): `other` 0.834, `bk` 0.063, `aa` 0.055, `cs` 0.027, `ar` 0.010, `co` 0.007, `cc` 0.004.

The control scores 0.271 against exp05's 0.298 on v4 (−0.027, within noise). The examples are drawn by index from each version's train split, so they may not be the same segments (seg_ids in each `settings.json`).

## Conclusion
**Two examples per label is the best prompt so far** (0.341, +0.070 over the control, twice the noise spread). The predictions come close to the real label distribution (`other` 13,151 against 13,634 gold; `other` F1 0.882), and `ar`, `cc` and `aa` all improve. The short replies are still the weak point: `bk` recall 0.08, with 702 of 1,029 gold `bk` predicted `other` and 241 predicted `aa`.

**The order of the examples matters, through the last one.** With `cs` last (reversed), `cs` predictions triple (744 → 2,327, 1,933 of them gold `other`) and `other` predictions halve: macro-F1 −0.057, beyond noise. The shuffled order (`co` last, `other` second) is within noise of the control (+0.007). This is the recency bias Zhao et al. (2021) describe.

**Calibration is not a lever here, in either form.** The paper's uniform target lowers every run, and the more the content-free probabilities lean to `other`, the more it costs: `other` 0.964 → −0.097 (control), 0.766 → −0.057 (2 per label), 0.465 → −0.036 (reversed), 0.077 → −0.018 (shuffled). Dividing out a lean to the majority class removes correct `other` predictions, as in exp03. Calibrating to the train frequencies avoids that and is 0.04–0.08 above uniform everywhere, but within noise of the raw scores (−0.018 to +0.020). It does not narrow the spread across orders either (raw 0.214–0.278, train frequencies 0.227–0.298), unlike the paper. Its datasets are near-balanced; ours is 84 % `other`, which is the most likely reason the gains do not carry over.

Limits: one draw of examples per number of examples (seed 0) and one shuffled order; data v3, while exp05–exp07 are on v4 (v3 and v4 differed by ≤ 0.002 in exp03 and exp04). Next: more examples per label (3 or more); with the cache, a 2-per-label run takes ~15 min.
