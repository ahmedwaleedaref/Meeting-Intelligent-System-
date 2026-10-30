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
pending

## Conclusion
pending
