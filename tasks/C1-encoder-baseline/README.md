# C1 — Encoder baseline (DeBERTa-v3-base) and hierarchical variant

## Goal
Produce the encoder baseline for 7-class dialogue-act tagging (val macro-F1 over the 6 target classes), and the hierarchical variant requested by the supervisor, as reportable protocol experiments.

Full description: [task.md](task.md). Built code and decisions so far: [DESIGN.md](../C-baselines/DESIGN.md), [REPORT.md](../C-baselines/REPORT.md).

## Status
In progress. The pipeline is built and trial-run on Kaggle.
- Flat model, seed 42, k = 0, fp32: val macro-F1 0.503.
- Hierarchical model, seed 42: 0.488.
- fp16 + fused AdamW make an epoch ~2× faster at batch 32.

Next: the experiments in task.md, run through `scripts/trial_run.py`.

## Decisions
- DeBERTa-v3-base, fine-tuned fully, weights loaded in fp32 — standard strong encoder; `transformers` 5 would otherwise load its stored fp16.
- k = 0 (target segment only) as the first baseline — every k runs through the same code.
- Plain cross-entropy first, class weighting as a separate experiment — measures what minority-class handling adds.
- Best epoch by val macro-F1 (6 targets), computed with the harness's own code — the number that selects is the number reported.
- fp16 + fused AdamW for new runs; `--fp32` reproduces the earlier setup — measured one change at a time, 10.1 → 4.9 min per epoch at batch 32.
- Batch size stays 32 until a val run shows 64 keeps the accuracy — batch size changes training, not just speed.
- Hierarchical model belongs to this task — it's a DeBERTa variant built on the same pipeline.

## Experiments

| Exp | Question | Result | Conclusion |
|-----|----------|--------|------------|
| [exp01](experiments/exp01_baseline-3-seeds/NOTE.md) | Flat baseline, 3 seeds (fp16 + fused, batch 32, k = 0): val macro-F1 and its seed spread | running | — |
