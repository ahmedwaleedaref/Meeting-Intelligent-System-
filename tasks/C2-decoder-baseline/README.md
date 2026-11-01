# C2 — Decoder + classification head baseline (Qwen, LoRA)

## Goal
A decoder baseline (a Qwen model with a 7-class classification head, fine-tuned with LoRA) on the same data, input and evaluation as the encoder baseline, plus a measured answer to whether LoRA handicaps the decoder on this task.

Full description: [task.md](task.md). Design: [DESIGN.md](../C-baselines/DESIGN.md), Part 5.

## Status
Not started. The shared pipeline is ready for it; the additions are mainly in model loading (task.md, part 2).

## Decisions
- Start with Qwen 0.5B, scale up one size at a time only if needed — the smallest model also allows the full fine-tuning control experiment on a T4.
- Same Dataset, Sampler, Collator and harness as the encoder — a family comparison must differ only in the model.
- LoRA for the baseline, with full fine-tuning at 0.5B as a control — LoRA trains ~1–2% of the parameters, so its cost must be measured, not assumed.

## Experiments

| Exp | Question | Result | Conclusion |
|-----|----------|--------|------------|
