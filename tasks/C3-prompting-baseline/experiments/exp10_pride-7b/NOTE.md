## Question
Does the 7B prefer some answer letters whatever option they stand for, and does dividing that preference out help? In every run so far A is always `cs` and G always `other`, so a preference for a letter and a preference for its class cannot be told apart. Contextual calibration (exp02, exp03, exp08) divided out both together and lost the model's mostly correct lean to `other`. PriDe (Zheng et al. 2024, "Large Language Models Are Not Robust Multiple Choice Selectors", ICLR 2024) estimates the letter part alone, by rotating which option stands behind which letter, without labels.

## Change vs baseline
Baselines, data v3: exp08's `c3_exp08_7b_shots2` (2 examples per label, seed 0, label order; macro-F1 0.341) and exp03's 7B zero-shot (`c3_exp03_7b`, 0.206 on v3). Both Qwen2.5-7B-Instruct 4-bit, k = 0, definitions on, `template_v1.txt`. Notebook `notebooks/kaggle_c3_exp10.ipynb`, dataset `smi-v3-meetings`.
- Each setting is scored under the 7 cyclic rotations of the options (`--option-shift 0..6`): the letters stay A–G in order and the option texts move up by the shift (shift 1: `A) command`, …, `F) other`, `G) suggestion`); the examples answer with their option's new letter. Shift 0 is the baseline prompt itself and should repeat 0.341 / 0.206, up to fp16 batching differences (the baselines ran with other batch settings).
- From the 7 runs, without the model (`scripts/pride_prompting.py`):
  - the letter prior over all segments (the paper's equation 7): how much each letter pulls, whatever it means;
  - **PriDe** (algorithm 1): the prior estimated on 5 % of the val segments (their scores only, no labels), which get the average of their rotations; the other 95 % get their shift-0 scores divided by the prior. 5 random draws of the 5 %, as in the paper;
  - **cyclic**: every segment's scores averaged over its 7 rotations (equation 1; 7 × the inference cost).
- Speed: `--prefix-cache`, batch 32, both T4s. Without the cache the 14 runs would take ~12 h on both GPUs, so the notebook stops if the cache check fails.

What the paper leads us to expect: over 20 LLMs, PriDe (5 %) adds +1.2 / +1.3 accuracy points (MMLU / ARC, 0-shot) and cyclic averaging +4.9 / +5.7; the large gains are for models with a strong letter bias. exp03 found no first-letter bias in the 7B, but it could not separate letters from classes.

## Runs
- v3: `c3_exp10_7b_shots2_r0` … `_r6`, `c3_exp10_7b_zs_r0` … `_r6`
- from them: `c3_exp10_7b_shots2_cyclic`, `c3_exp10_7b_shots2_pride0` … `_pride4`, and the same for `zs`

## Result
Commit `b3b0e2c`, notebook `notebooks/kaggle_c3_exp10.ipynb`, Kaggle 2 × T4, data v3 (val 16,289 segments). 13 of the 14 runs finished; `c3_exp10_7b_shots2_r5` stopped at its prefix-cache check (below), so the 2-per-label cyclic and PriDe results are missing. Speed 19–23 segments/s (2 per label), 25–30 (zero-shot). Mean letter mass ≥ 0.997 in every run.

Shift 0 repeats the baselines: 0.3410 (exp08: 0.341) and 0.2056 (exp03's uncached run: 0.2063; 98.6 % of the labels identical).

Every rotation (the layout shows where `other` and `bk` stand):

| shift | `other` / `bk` at | zero-shot macro-F1 | predicted `other` / `bk` | 2 per label macro-F1 | predicted `other` / `bk` |
|---|---|---|---|---|---|
| 0 (baseline) | G / D | 0.206 | 8,968 / 5,893 | **0.341** | 13,151 / 168 |
| 1 | F / C | **0.316** | 1,816 / 11,631 | 0.317 | 5,347 / 7,811 |
| 2 | E / B | 0.277 | 3,653 / 10,768 | 0.250 | 3,020 / 12,595 |
| 3 | D / A | 0.234 | 10,271 / 3,934 | 0.318 | 7,264 / 3,139 |
| 4 | C / G | 0.241 | 14,653 / 709 | 0.223 | 15,840 / 16 |
| 5 | B / F | 0.192 | 9,004 / 6,169 | (stopped) | |
| 6 | A / E | 0.184 | 8,397 / 7,348 | 0.270 | 10,622 / 5,158 |

(gold `other` 13,634, `bk` 1,029)

Zero-shot, from the 7 rotations:

| zero-shot, v3 | macro-F1 | cs | co | aa | bk | ar | cc | predicted `other` / `bk` |
|---|---|---|---|---|---|---|---|---|
| shift 0 (baseline) | 0.206 | 0.374 | 0.137 | 0.043 | 0.218 | 0.050 | 0.412 | 8,968 / 5,893 |
| **cyclic** (average of the 7 rotations) | **0.297** | 0.401 | 0.125 | 0.034 | 0.243 | 0.561 | 0.418 | 8,655 / 6,411 |
| PriDe, 5 draws of 5 % (814 segments) | 0.187 / 0.189 / 0.182 / 0.188 / 0.183 (mean 0.186) | 0.39–0.40 | 0.13–0.14 | 0.03–0.04 | 0.15 | 0.07–0.10 | 0.31–0.32 | ~3,150 / ~11,750 |

The cyclic gain is mostly `ar` (F1 0.050 → 0.561): at shift 0 `ar` is predicted 50 times (gold 149); averaged over the orders, 204.

Letter prior (equation 7), over all segments: A 0.189, B 0.098, C 0.062, D 0.015, E 0.017, F 0.016, **G 0.603**; the five 5 % draws give G 0.596–0.618, so the estimate is stable. But G is not what the model picks: over the 7 rotations it picks A 12.7 %, B 19.6 %, C 24.5 %, D 14.9 %, E 10.7 %, F 8.3 %, G 9.2 % of the time. The prior comes from the tail. When the last letter shows an option the model rejects (anything but `other` and `bk`), its median probability is still 1e-6 to 2e-3 (mean log probability −7 to −14). The other letters showing a rejected option get 1e-9 to 3e-5 (−10 to −20). Equation 7 averages log probabilities, so this floor dominates. Dividing the prior out of the shift-0 scores cuts `other` (at G) about 40 × against `bk` (at D: 0.603 / 0.015): `other` predictions fall from 8,968 to ~3,150, `bk` predictions rise to ~11,750.

The failed run: `shots2_r5`'s check measured cached 0.024 against padded batch 0.00011, allowed 0.010 (3 × padded + 0.01). In the 19 checks that passed so far (exp08–exp10), cached ranged 0.0001–0.179. The padded reference was unusually small here, probably because the 8 checked prompts needed no padding. The 0.024 itself is ordinary fp16 noise: the zero-shot shift 0 (cache, batch 32) against exp03's run (no cache, batch 16) differs by more than 0.024 on 3,201 segments (99th percentile 0.078, max 0.271), with macro-F1 0.2056 vs 0.2063. So the check's fixed allowance was raised from 0.01 to 0.05 (`CHECK_FLOOR` in `src/smi/prompting/backend.py`), after this failure and for this reason, and shift 5 is run again alone (`notebooks/kaggle_c3_exp10_r5.ipynb`, same settings as the other six). The 2-per-label averaging and PriDe are then computed from the seven runs' scores, without the model.

## Conclusion
**The order of the options matters far beyond prompt noise.** Rotating the same 7 options moves zero-shot macro-F1 between 0.184 and 0.316 (spread 0.132), and with 2 examples per label between 0.223 and 0.341 (six of the seven orders, spread 0.118); the prompt-noise spread is 0.036 (exp07). This is the sensitivity Zheng et al. (2024) report. Here it is not a simple preference for a letter: the top answer follows the content (`other` or `bk` wins most segments wherever they stand), but where they stand moves thousands of segments between the two (`bk` is predicted 709 times when it is last, 11,631 times when it is third). The best single order cannot be chosen on val without tuning on it; shift 0 happens to be the best order with examples and among the worst without.

**Averaging over the 7 orders is the gain**: zero-shot 0.206 → 0.297 (+0.091, as much as one example per label gave in exp05), with no labels and no examples, at 7 × the inference. The paper also finds the averaging itself (+4.9 / +5.7 accuracy) worth more than PriDe.

**PriDe lowers the score** (mean 0.186, −0.020; all five draws below the baseline). Its prior assumes a letter multiplies every option's probability by the same factor. In this model the last letter raises the floor of rejected options far more than it raises the chosen ones, and the log-average turns that floor into a 0.60 prior. Dividing it out removes correct `other` predictions, the same way calibration did (exp02, exp03, exp08).

Pending: the 2-per-label cyclic and PriDe results, which need shift 5.

Limits: one run per order; data v3. PriDe's prior is estimated on val segments without their labels, as the paper does on test inputs.
