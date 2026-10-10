# B — Evaluation harness

## Goal
One evaluation function that scores a prediction file against the frozen dataset (A10),
independent of which model produced it. Current scope: layer 1 (utterance classification) only.
Success = `evaluate(pred_path, split)` writes a correct `metrics.json`, verified by the tests listed in [task.md](task.md#b2-layer-1-metrics).

Full task description (schema, validation, metrics, output layout, tests): [task.md](task.md).

## Status
Layer-1 harness done for the current scope: prediction schema (B1), metrics (B2) and tests (B3). Next: `evaluate()` (load predictions + gold, test-split flag, write `metrics.json`), written in the baseline phase once `scripts/run.py` exists.

| Sub-task | Status |
|---|---|
| B1 Prediction schema | done: coverage enforced by the seg_id check in the metrics |
| B2 Layer-1 metrics | metrics done; `evaluate()` deferred to the baseline phase |
| B3 Tests | done |
| B4 Layer-2 metrics | deferred: linking only matters once utterances are classified well |
| B5 Uncertainty (bootstrap) | deferred: needs formalisation first (Koehn 2004; Dror et al. 2018) |
| B6 CV runner | deferred: only if the standard split proves too noisy |
| B7 Results log | deferred to the baseline phase |

Roles: design + implementation (owner), tests (one member, written from task.md), code review (one member, §5 of PROTOCOL).

## Decisions
- Prediction format is JSONL, one line per segment, one file per (run, split) — one format serves every model family.
- Join key is `seg_id` (column 2 of the `.dadb` file); matching never uses line order — robust to reordering and filtering.
- Segment position in its meeting is a separate dataset field, never part of `seg_id` — position changes when filtering changes.
- The 7-class label set is defined once in `src/smi/labels.py` and imported everywhere — single source of truth.
- Prediction files contain no gold labels; the harness reads gold from the frozen dataset — prevents leakage and stale gold.
- `scores` required for every model that can produce them; `null` only for generation-only prompting — needed for PR-AUC and calibration.
- Validation failures raise an error, never a warning — a silently wrong metric is worse than a crash.
- Primary metric = macro-F1 over the 6 target classes, excluding `other` — `other` dominates and would hide target-class performance.
- 3-class level and `cs` + `co` pooled dropped from B2 for now — the 3-class representation will most likely not be kept.
- Harness is threshold-free (argmax); threshold tuning, if any, is an experiment — keeps the harness model-agnostic.
- `zero_division=0`: a class with no predictions gets precision 0, never NaN — keeps macro averages defined.
- Evaluating `test` requires an explicit flag recorded in the output — PROTOCOL §7.
- `seg_id` uniqueness is guaranteed by the data side (task A); the harness only rejects duplicates inside a prediction file — the harness checks files, not the corpus.
- A prediction file is all-scores or all-null, never mixed; only prompting outputs may be null — a mixed file means a broken pipeline.
- Coverage (missing / extra seg_ids vs the split) is enforced by `per_class_metrics` / `confusion_matrix`, which raise if prediction and gold IDs differ — no separate check needed.
- A target class with zero gold support counts as F1 = 0; macro-F1 always divides by the full class count (6 or 7) — a class is never silently dropped.
- PR-AUC deferred until the first score-producing model exists — nothing to compute it on yet.
- `evaluate()` deferred to the baseline phase — its interface depends on how `run.py` calls it.

### Open
- [ ] Must `label` agree with argmax of `scores`? Not enforced for now; revisit with the first baseline outputs.
- [ ] Source of `data_version` in `metrics.json`: proposed: `source.json` of the per-meeting files (copied from the release metadata). Settle with `evaluate()`.

## Experiments
The harness is shared code (`src/smi/eval/`), so this task is not expected to have experiments.

| Exp | Question | Result | Conclusion |
|-----|----------|--------|------------|
