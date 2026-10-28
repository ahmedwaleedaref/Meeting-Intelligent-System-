## Question
Do more examples keep helping? exp08 found 2 solved examples per label clearly better than 1 (0.341 vs 0.271). Does 3 per label improve further, and how much of a few-shot score is just the particular examples drawn?

## Change vs baseline
Baseline: exp08's `c3_exp08_7b_shots2` (Qwen2.5-7B-Instruct 4-bit, k = 0, definitions on, `template_v1.txt`, 2 examples per label, seed 0, label order; val macro-F1 0.341). Data v3, as exp08. Notebook `notebooks/kaggle_c3_exp09.ipynb`, dataset `smi-v3-meetings`.
- Only change: `--shots-per-class 3` (21 examples, label order, so `other` stays last as in the baseline). A different draw from the baseline's, not its 14 examples plus 7.
- Second run, on the otherwise idle GPU: the same with `--shot-seed 1`, other examples. The gap between the two seeds is the spread due to the draw alone, which exp05 and exp08 could not measure.
- Not calibrated: exp08 found calibration does not help in either form.
- Speed as in exp08: `--prefix-cache` (checked on the 7B first; without it the runs go at batch 16), batch 32, one run per T4.

## Runs
- v3: `c3_exp09_7b_shots3` (seed 0), `c3_exp09_7b_shots3_seed1`

## Result
Commit `d78b088`, notebook `notebooks/kaggle_c3_exp09.ipynb`, Kaggle 2 × T4, data v3 (val 16,289 segments). The cache passed its check and was used: 739 / 725 shared tokens (seed 0 / seed 1); largest difference from each full prompt scored alone: cached 0.060 / 0.113, padded batch 0.060 / 0.091. Speed 20.9 / 18.5 segments/s, ~13 / ~15 min per run. Mean letter mass 0.995 / 0.998 (below 0.5 for 14 / 2 segments).

The examples: 3 of the baseline's 14 are among seed 0's 21 (two `cs`, one `co`); seed 1 shares none with either.

| 7B, k = 0, v3, label order | seed | macro-F1 | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|---|---|
| 2 per label (exp08, baseline) | 0 | **0.341** | 0.372 | 0.215 | 0.376 | 0.139 | 0.548 | 0.397 | 0.882 |
| 3 per label | 0 | 0.251 | 0.100 | 0.035 | 0.383 | 0.344 | 0.426 | 0.215 | 0.407 |
| 3 per label | 1 | 0.297 | 0.224 | 0.152 | 0.395 | 0.128 | 0.566 | 0.317 | 0.813 |

(per-class F1; macro-F1 over the 6 targets, without `other`)

Predicted (baseline / seed 0 / seed 1): `cs` 558 / 7,558 / 2,672 (gold 454); `bk` 168 / 3,019 / 82 (gold 1,029); `aa` 2,131 / 1,962 / 2,046 (gold 824); `other` 13,151 / 3,624 / 11,281 (gold 13,634).

Where the errors go:
- `cs`: seed 0 predicts it for 6,792 gold `other` segments (`cs` recall 0.881, precision 0.053), seed 1 for 2,201. The segments seed 0 calls `cs` average 9.6 tokens (words and punctuation marks), close to gold `other` (9.1) rather than gold `cs` (16.9): it labels ordinary statements `cs` (2,999 of the 5,912 gold `other` segments with ≥ 8 tokens). Seed 1's and the baseline's average 17.5 / 17.2.
- At k = 0 the prompt holds the segment's text only, so identical texts get identical answers. "OK" alone (608 val segments: gold 346 `bk`, 254 `other`) is answered `other` for all 608 by the baseline and seed 1, and `bk` for all 608 by seed 0. These are 346 of seed 0's 697 correct `bk`, most of its `bk` F1 jump (0.139 → 0.344). "Yeah" alone (1,473 segments: gold 939 `other`, 374 `aa`, 158 `bk`) is answered `aa` for at least 1,451 by all three runs: the `aa` over-prediction.
- The examples drawn as `other`: seed 0 "Say it again ?", a short technical statement, "that ho that whole th"; seed 1 "OK .", "and , uh , someth", "there 's a potential .". The baseline's include a long opinion statement. Much of gold `other` is longer talk (5,912 of its 13,634 segments have ≥ 8 tokens); both 3-per-label draws show mostly short or cut-off `other` examples.

## Conclusion
**Three examples per label did not beat two**: both draws are below the baseline (0.251 and 0.297 vs 0.341, −0.090 and −0.044). Since the same seed draws mostly different segments for 3 per label than for 2, this does not show that a third example hurts. It shows that the number of examples alone does not decide the score.

**Which examples are drawn moves the score beyond prompt noise.** Two draws of the same setting differ by 0.046 macro-F1, above the prompt-noise spread (0.036, exp07), and per class by far more: `other` F1 0.407 vs 0.813, `bk` 0.344 vs 0.128, `cs` predicted 7,558 vs 2,672 times. At k = 0 the model follows the surface of the particular examples: whether "OK" means `bk` or `other` flips all 608 such segments at once, and seed 0's examples lead it to call ordinary statements `cs`. The likely reason, not tested here, is that the random `other` examples do not look like much of the class (longer statements).

So exp08's 0.341 is one draw, and its +0.070 over one example per label mixes the number of examples with the luck of the draw. Few-shot still beats zero-shot (every label-order run so far: 0.251–0.341, zero-shot 0.207 on v4), but by an amount that depends on which examples are shown.

Limits: two draws at 3 per label, one at 2; data v3. Next: choose the examples instead of drawing them at random (e.g. `other` examples that are statements, plus one short reply), or score several draws at 2 per label to see whether 0.341 holds.
