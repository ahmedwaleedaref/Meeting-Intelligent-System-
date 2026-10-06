# B — Evaluation harness

## Goal
One evaluation function that scores a prediction file against the frozen dataset (A10),
independent of which model produced it. Current scope: layer 1 (utterance classification) only.
Success = `evaluate(pred_path, split)` writes a correct `metrics.json`, verified by the tests listed in [task.md](task.md#b2-layer-1-metrics).

Full task description (schema, validation, metrics, output layout, tests): [task.md](task.md).

## Status
In progress. Next step: owner proposes the code layout and function signatures (the Deliverables in task.md); team reviews them before implementation.

| Sub-task | Status |
|---|---|
| B1 Prediction schema | in progress |
| B2 Layer-1 metrics | in progress |
| B3 Layer-2 metrics | deferred: linking only matters once utterances are classified well |
| B4 Uncertainty (bootstrap) | deferred: needs formalisation first (Koehn 2004; Dror et al. 2018) |
| B5 CV runner | deferred: only if the standard split proves too noisy |
| B6 Results log | deferred to the baseline phase |
| B7 Error analysis | removed: becomes its own task |

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
- Coverage check against the split's gold IDs (missing / extra) deferred until the gold loader exists (A10) — nothing to check against yet.

### Open
- [ ] Must `label` agree with argmax of `scores`? Not enforced for now; revisit with the first baseline outputs.
- [ ] Target class with zero gold support in a split: counts as F1 = 0 in macro-F1 (6), or excluded? Proposed: count it as 0 and report support.
- [ ] PR-AUC from day 1 or later? Proposed: day 1 (`average_precision_score`), `null` when no scores or no positives.
- [ ] Source of `data_version` in `metrics.json`: proposed: frozen dataset metadata, not a function argument.

## Experiments
The harness is shared code (`src/smi/eval/`), so this task is not expected to have experiments.

| Exp | Question | Result | Conclusion |
|-----|----------|--------|------------|
