# B. Evaluation harness

Scope for now: utterance classification only (layer 1). The harness evaluates a prediction file against the frozen dataset. It does not know or care which model produced the file.

| Task | Status |
|---|---|
| B1 Prediction schema | done |
| B2 Layer-1 metrics | metrics done; `evaluate()` deferred to the baseline phase |
| B3 Tests | done |
| B4 Layer-2 metrics | deferred: linking only matters once utterances are classified well |
| B5 Uncertainty (bootstrap) | deferred: needs formalisation first |
| B6 CV runner | deferred: only if the standard split proves too noisy |
| B7 Results log | deferred to the baseline phase |

---

## B1. Prediction schema

**Goal:** one file format that every model family (encoder, decoder + head, prompting) must produce, so one evaluation function serves all of them.

**Format:** JSONL, one line per segment, one file per (run, split):

```json
{"seg_id": "Bdb001-c1_0479178_0479998", "label": "cs",
 "scores": {"cs": 0.71, "co": 0.04, "aa": 0.02, "bk": 0.01, "ar": 0.00, "cc": 0.05, "other": 0.17}}
```

| Field | Meaning |
|---|---|
| `seg_id` | Segment ID from column 2 of the `.dadb` file (meeting + channel + times). The join key with the gold labels. |
| `label` | Final predicted class, one of the 7. |
| `scores` | Probability per class, all 7 keys, summing to ≈1. `null` allowed only for generation-only prompting. |

**Rules:**
- The prediction file contains no gold labels. The harness reads gold from the frozen dataset (A10) by `seg_id`.
- Matching is by `seg_id`, never by line order.
- The position of a segment in its meeting is a separate dataset field, never part of the ID (it changes when filtering changes).
- The 7-class label set is defined once, in `src/smi/labels.py`, and imported by data, models and eval.

**Validation** (the harness must fail loudly, not warn):
- every `seg_id` of the evaluated split appears exactly once: no missing, no duplicates, no extras;
- every `label` is in the label set;
- if `scores` is present: exactly the 7 keys, values in [0, 1], sum within 1 ± 1e-3.

**Deliverables:** to be designed by the task owner.

**Decisions:**
- [x] Join key = segment ID from column 2.
- [x] Scores required for every model that can produce them; `null` only for generation-only prompting.
- [ ] Confirm `seg_id` is unique across the corpus (check in A1; if not, add a suffix and document it).

---

## B2. Layer-1 metrics

**Goal:** `evaluate(pred_path, split) -> dict`, called by `run.py` and by the prompting pipeline, writing `metrics.json`.

**Outputs:**

| Metric | Detail |
|---|---|
| Per-class P / R / F1 / support | all 7 classes |
| Primary: macro-F1 (6 targets) | mean F1 over `cs, co, aa, bk, ar, cc`, excluding `other` |
| Macro-F1 (7 classes) | secondary, for reference |
| Confusion matrix | 7×7, raw counts and row-normalised (recall view) |
| PR-AUC per class | only when `scores` is present |

**Rules:**
- Argmax only. The harness never tunes thresholds. Threshold tuning, if done, is an experiment.
- A class with zero predictions gets precision = 0 (no NaN), i.e. `zero_division=0`.
- The harness reports which split it evaluated. Evaluating `test` requires an explicit flag, which is recorded in the output (see PROTOCOL §7).

**`metrics.json` layout:**

```json
{"split": "val", "data_version": "v1", "n_segments": 17643,
 "primary": {"macro_f1_6": 0.00},
 "per_class": {"cs": {"p": 0.0, "r": 0.0, "f1": 0.0, "support": 0}, "...": {}},
 "macro_f1_7": 0.00,
 "confusion": {"labels": [], "counts": [[]]},
 "pr_auc": null}
```

**Tests:** see B3.

**Deliverables:** to be designed by the task owner.

**Decisions:**
- [x] Primary metric = macro-F1 over the 6 target classes, excluding `other`.
- [x] Drop the 3-class level and `cs` + `co` pooled for now: the 3-class representation will most likely not be kept.
- [x] Harness is threshold-free (argmax).
- [x] PR-AUC deferred until the first score-producing model exists.

---

## B3. Tests

**Goal:** prove the harness (B1 + B2) is correct before anyone uses its numbers. Tests are written from this document, not from the code.

**Rules:**
- Each test is described here first (what it checks, input, expected result), then implemented.
- Tests live in `tests/eval/` and run with `PYTHONPATH=src pytest tests/eval`.
- Tests that need the frozen release skip, never fail, when the data is not available locally.

**Tests** (required before anyone uses the harness):
- [x] perfect predictions → every F1 = 1.0;
- [x] all `other` → macro-F1 (6) = 0.0;
- [x] toy oracle: real segments from 3 meetings, labels changed by hand, P/R/F1 computed by hand → exact expected values (toy file and oracle merged);
- [x] every number cross-checked against `sklearn.metrics` (synthetic and real splits);

**Test descriptions:**

**T1. Perfect predictions.** A sanity check for major bugs in the metrics code.
- Input: gold labels loaded with `load_gold`, used as the predictions too (each wrapped as `{"label": ..., "scores": None}`).
- Expected: P = R = F1 = 1.0 for every class that occurs in the gold; a class with zero support gets 0 (by the zero-support decision), not 1.
- Note: one meeting does not always contain all 7 classes; every split does.

**T2. All `other`.** A model that always predicts `other` must score 0 on the primary metric.
- Input: gold labels loaded with `load_gold`; every prediction set to `other`.
- Expected: macro-F1 (6) = 0.0, and P = R = F1 = 0 for each of the 6 target classes.

**T3. Toy oracle** (toy file and oracle test merged). Catches subtle bugs that T1/T2 cannot (swapped P/R, wrong denominators, macro as a sum).
- Input: `tests/fixtures/toy/`: 3 val meetings × 10 consecutive real segments (all 7 classes present) as gold, and the same 30 segments as a prediction file with 5 labels changed by hand.
- Expected: per-class P/R/F1/support, macro-F1 (6) and macro-F1 (7) equal the values computed by hand in `tests/fixtures/toy/README.md`.
- Edits cover: `ar` never predicted (zero division), cs → co, gold `other` predicted as a target, wrong target. Not covered: a target predicted as `other` (owner's choice).
- Fixtures are in git, so the test always runs.

**T4. sklearn cross-check.** Every number the harness produces must equal `sklearn.metrics` (with `zero_division=0`).
- Compared: per-class P/R/F1/support, macro-F1 (6) and (7), confusion matrix, row-normalised confusion matrix.
- Synthetic (always runs): random MRDA-like gold, 15 and 500 segments × 3 seeds; the 15-segment runs leave classes absent (zero-division paths).
- Real (skipped without `data/v4_meetings`): `train` and `val` gold, predictions made from gold with a fixed seed at 30 %, 60 % and 90 % accuracy.
- scikit-learn is a required dependency; a missing install fails the tests instead of skipping them.

---

## B4. Layer-2 metrics — deferred

Per-proposal link accuracy and pair-level P/R/F1. Revisit once layer 1 works.

## B5. Uncertainty — deferred

Meeting-level (cluster) bootstrap CIs and paired bootstrap for model comparisons. Before implementing: formalise the method (Koehn 2004; Dror et al. 2018), and check empirically that the intervals are not too narrow with only 12 clusters.

## B6. CV runner — deferred

Grouped K-fold over train + val meetings, out-of-fold predictions, test only in final mode. Trigger: adopt it if results on the standard split prove too noisy to decide between configs, e.g. seed-to-seed spread on validation for `cc`/`co` is as large as the differences being compared.

## B7. Results log — deferred to the baseline phase

Tool choice (W&B vs CSV + collect script), seeds per config.
