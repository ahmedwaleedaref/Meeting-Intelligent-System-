# C3 — Prompting baseline

## Goal
A prompting baseline (an LLM classifying each segment from a prompt, without fine-tuning) on the same data and evaluation as the encoder and decoder, with scores from label log-probabilities where the backend allows it.

Full description: [task.md](task.md). Design: [DESIGN.md](../C-baselines/DESIGN.md), Part 6.

## Status
Not started. The Dataset and the harness are ready for it; the prompt renderer, the backends and the cache are to be written. Open: the API budget, and which larger model to use for the scale runs and where to run it (task.md, "Open, decide first").

## Decisions
- Instruct checkpoints, inputs formatted with each model's own chat template (`apply_chat_template`); no instruction tuning or other training of our own — the makers' instruction tuning is what makes the model follow the prompt.
- Scores from the 7 letters' log-probabilities wherever the backend allows it, generation only as the fallback — probabilities allow tuning the `other`-vs-target trade-off on val and give the same evaluation as the fine-tuned models.
- Letters A–G plus descriptive names as answer options, never raw MRDA codes — each answer is one token, so it can be scored by log-probability.
- `other` always defined in the prompt — it's the majority class; without a definition the model over-predicts the targets.
- Unparseable answers fall back to `other`, with the failure rate recorded — `other` is not a target class, so the fallback adds no false positives to the primary metric.
- Few-shot examples stratified (1–2 per class), from train only, with a fixed seed and a fixed order, the same in every cell — every class is shown, PROTOCOL §7, and one change per comparison. Retrieval of similar examples is a later experiment.

## Experiments

| Exp | Question | Result | Conclusion |
|-----|----------|--------|------------|
