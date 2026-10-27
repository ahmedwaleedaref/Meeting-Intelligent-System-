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
- lr 1e-5, 3 epochs (exp02 grid, confirmed with 3 seeds in exp03) — 3 epochs won at every LR, 1e-5 was the stable one; whether it's also best for k > 0 is its own later experiment.
- Gradient checkpointing only where batch 32 doesn't fit (k = 10; k = 5 if its memory check fails) — the same gradients bit for bit, so runs with and without it compare directly; costs ~1/3 more time per step.
- Hierarchical model belongs to this task — it's a DeBERTa variant built on the same pipeline.

## Experiments

| Exp | Question | Result | Conclusion |
|-----|----------|--------|------------|
| [exp01](experiments/exp01_baseline-3-seeds/NOTE.md) | Flat baseline, 3 seeds (fp16 + fused, batch 32, k = 0): val macro-F1 and its seed spread | 0.499 ± 0.004 (seeds 42/43/44: 0.4985 / 0.5032 / 0.4949); ~27 min per seed | the reference for C1; differences < ~0.01 are seed noise; two parallel runs don't slow each other |
| [exp02](experiments/exp02_lr-epochs-grid/NOTE.md) | LR {1e-5, 2e-5, 3e-5, 5e-5} × epochs {3, 5, 8}, k = 0, seed 42: the best setting to fix for the context experiments | 3 epochs best at every LR (0.517–0.525; 5 epochs 0.498–0.513; 8 epochs 0.488–0.500); best cells 1e-5 / 5e-5 at 3 epochs (0.5250 / 0.5254) | epochs matter, the LR doesn't within one seed's noise; the top 2 cells go to exp03 |
| [exp03](experiments/exp03_confirm-top-cells/NOTE.md) | Confirm (1e-5, 3 epochs) and (5e-5, 3 epochs) with seeds 43 and 44 | 1e-5: 0.5225 ± 0.0059; 5e-5: 0.5105 ± 0.0142 (3 seeds each) | adopted lr 1e-5, 3 epochs; new k = 0 reference 0.5225 (+0.024 over exp01); 5e-5's exp02 score was a lucky seed |
| [exp04](experiments/exp04_context-k3/NOTE.md) | Context k = 3 (separator `"\n"`), adopted setting, 3 seeds: does context help? | 0.5524 ± 0.0103 (seeds 42/43/44: 0.5419 / 0.5624 / 0.5529); ~31 min per seed | context helps: +0.030, every seed improves; mostly `aa` (+0.144, fewer `aa` predicted as `other`); next k = 5, 10 |
| [exp05](experiments/exp05_context-k5/NOTE.md) | Context k = 5 (separator `"\n"`), adopted setting, 3 seeds: does more context help further? | pending | pending |
| [exp06](experiments/exp06_context-k10/NOTE.md) | Context k = 10 (separator `"\n"`, gradient checkpointing), adopted setting, 3 seeds: does a much wider context help? | pending | pending |
