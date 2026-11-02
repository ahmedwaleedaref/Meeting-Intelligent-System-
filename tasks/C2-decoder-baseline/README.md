# C2 — Decoder + classification head baseline (Qwen, LoRA)

## Goal
A decoder baseline (a Qwen model with a 7-class classification head, fine-tuned with LoRA) on the same data, input and evaluation as the encoder baseline, plus a measured answer to whether LoRA handicaps the decoder on this task.

Full description: [task.md](task.md). Design: [DESIGN.md](../C-baselines/DESIGN.md), Part 5.

## Status
Experiment done; the review remains. Deliverables (task.md): the [Part 1 note](part1-note.md); the code in `src/` and `scripts/trial_run.py` with `tests/models/test_decoder.py`; exp01 below, including the recommendation. The encoder path is unchanged, byte for byte on a T4 (`notebooks/kaggle_c2_default_path_check.ipynb`). All runs are trial runs of the temporary driver (PROTOCOL §4.1), on val; test was never read.

**Result:** Qwen2.5-0.5B with LoRA r = 16 reaches val macro-F1 (6) **0.4807 ± 0.0102**. LoRA costs nothing measurable against full fine-tuning (0.4728 ± 0.0122), and all linear layers add nothing (0.4793 ± 0.0130). The decoder is behind the encoder (0.5225 ± 0.0059) by 0.042, on `cc`, `co` and `cs`. Every decoder result: 2,168,960 trainable parameters (0.439 %), r = 16, α = 32, attention projections, LoRA; LoRA vs full gap +0.0079, not distinguishable from 0.

**Next:** a reviewed PR for the code (PROTOCOL §5). Larger models and k > 0 aren't required by task.md: the open choice is below.

## Decisions
- Start with Qwen 0.5B, scale up one size at a time only if needed — the smallest model also allows the full fine-tuning control experiment on a T4.
- Same Dataset, Sampler, Collator and harness as the encoder — a family comparison must differ only in the model.
- LoRA for the baseline, with full fine-tuning at 0.5B as a control — r = 16 on attention trains 0.44 % of the parameters (1.78 % on all linear layers), so its cost must be measured, not assumed.
- `Qwen/Qwen2.5-0.5B`, the base checkpoint, not Instruct — the head replaces the language-model output, so instruction tuning buys nothing here.
- Suffix `"
Label of [T]:"`, r = 16, α = 32, dropout 0.05, fixed across decoder runs — a model comparison must not be a settings comparison.
- The model always gets `tokenizer.pad_token_id`, never `eos_token_id` — the two differ for Instruct and Qwen3, and a mismatch is silently wrong.
- Frozen base in fp16, trainables in fp32, checked at load — `GradScaler` refuses fp16 gradients, and PEFT leaves the head in the base's dtype.
- A collator with a suffix raises on truncation — truncation cuts from the end, which is the token the head reads.
- The capacity run is all linear layers at r = 16, not r = 64 — about the same parameter count, and it leaves r and α alone.
- 3 settings × seeds 42, 43, 44, with no LR search — the Kaggle budget; LoRA lr 2e-4, full lr 1e-5, and the note says what that leaves open.
- LoRA r = 16 on the attention projections is the decoder default, fixed across sizes (exp01) — within noise of full fine-tuning and of all linear layers, and the cheapest of the three.
- LoRA is a fair stand-in for larger models at k = 0 (exp01) — re-check the gap at the first k > 0 setting, because k = 0 caps `aa` and `bk` for every method.
- Gradient checkpointing for full fine-tuning at 0.5B — batch 32 ran out of memory without it; the same batches and gradients, so the arms compare directly.
- A decoder difference needs to be above ~0.02 to count at 3 seeds — the decoder's own seed spread (std 0.010–0.013), twice the encoder's.

## Experiments

| Exp | Question | Result | Conclusion |
|-----|----------|--------|------------|
| [exp01](experiments/exp01_lora-vs-full/NOTE.md) | LoRA r = 16 vs LoRA all linear layers vs full fine-tuning on Qwen2.5-0.5B, k = 0, 3 seeds: are we handicapping the decoder? | LoRA r16 0.4807 ± 0.0102; all linear 0.4793 ± 0.0130; full 0.4728 ± 0.0122; ~21 / 29 / 41 min per run | no handicap: LoRA ≈ full, no capacity limit, LoRA r16 stays the default; the decoder is 0.042 behind the encoder, on `cc`, `co`, `cs` |
