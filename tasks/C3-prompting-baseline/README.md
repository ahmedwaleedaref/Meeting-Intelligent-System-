# C3 — Prompting baseline

## Goal
A prompting baseline (an LLM classifying each segment from a prompt, without fine-tuning) on the same data and evaluation as the encoder and decoder, with scores from label log-probabilities where the backend allows it.

Full description: [task.md](task.md). Design: [DESIGN.md](../C-baselines/DESIGN.md), Part 6.

## Status
In progress. Built and tested: prompt files (`configs/prompts/`), the renderer, few-shot selection and the local letter-scoring backend (`src/smi/prompting/`), and the temporary driver `scripts/trial_prompting.py` (resumable, writes `predictions.jsonl` and `metrics.json` through the harness). Next: the first full val run on Kaggle (zero-shot, k = 0). Not built yet: an API backend and the generation fallback (no API budget). Open: the API budget, and which larger model to use for the scale runs and where to run it (task.md, "Open, decide first").

Data: release **v4** (`data/v4_meetings`, val 16,188 segments), the same data as the encoder (C1). exp01–exp03 were first run on v3 while v4 was not available locally; their results below are those v3 trial runs, and they are rerun on v4 before they count. Every `metrics.json` records the data version.

## Decisions
- Instruct checkpoints, inputs formatted with each model's own chat template (`apply_chat_template`); no instruction tuning or other training of our own — the makers' instruction tuning is what makes the model follow the prompt.
- Scores from the 7 letters' log-probabilities wherever the backend allows it, generation only as the fallback — probabilities allow tuning the `other`-vs-target trade-off on val and give the same evaluation as the fine-tuned models.
- Letters A–G plus descriptive names as answer options, never raw MRDA codes — each answer is one token, so it can be scored by log-probability.
- `other` always defined in the prompt — it's the majority class; without a definition the model over-predicts the targets.
- Unparseable answers fall back to `other`, with the failure rate recorded — `other` is not a target class, so the fallback adds no false positives to the primary metric.
- Few-shot examples stratified (1–2 per class), from train only, with a fixed seed and a fixed order, the same in every cell — every class is shown, PROTOCOL §7, and one change per comparison. Retrieval of similar examples is a later experiment.

Implementation:
- Main model `Qwen/Qwen2.5-0.5B-Instruct` — the instruct twin of C2's Qwen 0.5B, so prompted vs fine-tuned is compared on the same model.
- Letters follow the `LABELS` order (A = cs … G = other) — fixed across runs. In Qwen's tokenizer each is one token (ids 32–38, checked at load time); the model's top next tokens after the opened assistant turn are these letters without a space (letter mass ≈ 1.0 on real val prompts).
- Scoring: one forward pass per prompt; only the last position's logits are computed (`logits_to_keep=1`); `scores` = softmax over the 7 letter logits in fp32 — every segment gets a label and 7 scores, and the same prompt always gives the same numbers.
- Batches are padded on the left, with positions counted over real tokens only — a padded prompt gets the same scores as the prompt alone (tested).
- `letter_mass` (share of the full next-token probability on A–G) is recorded per segment, its mean and minimum in `metrics.json` — a low value means the model ignores the answer format and its scores are unreliable.
- fp16 on the GPU (no bf16 on a T4) with a NaN/inf check that stops the run, fp32 on CPU; optional 4-bit loading for models too large for a T4 in fp16.
- Prompts are files in `configs/prompts/` (`template_v1.txt`, `definitions_v1.txt`, `names_v1.txt`); a changed prompt is a new file, never an edit; the sha256 of the template and options files is recorded in `settings.json` — every run can be traced to the exact text it used.
- Definitions are paraphrased from the MRDA labelling manual (Dhillon et al. 2004, the sections Accept, Reject, Acknowledgment, Command, Suggestion, Commitment) — the model is judged against the annotators' rules, so it gets the same rules.
- `other` is defined to include plain backchannels ("uhhuh" while someone else is talking) — the project maps the MRDA backchannel tag `b` to `other`, and the manual names `b` / `bk` / `aa` as the most confused tags.
- Few-shot examples are drawn per class with Python's `random.Random(seed)`, in `LABELS` order, with the same context window as the question; their seg_ids go into `settings.json` — the same seed gives the same segments at every k.
- One prompt for everything, with the transcript last — instructions are read before the data; definitions on/off swaps only the options block, so the two variants differ in nothing else.
- Each run writes every segment's scores to `scores.jsonl` as it goes; rerunning the same command skips scored segments, and a different setting under the same run name is refused — survives Kaggle's 12 h cap. Local scoring is deterministic, so no cross-run response cache is needed until an API backend exists.

## Experiments

| Exp | Question | Result | Conclusion |
|-----|----------|--------|------------|
| [exp01](experiments/exp01_zero-shot-k0/NOTE.md) | Zero-shot, k = 0, definitions on, Qwen2.5-0.5B-Instruct: the reference point for the grid | v3 (superseded, rerun on v4 pending): macro-F1 0.019; letter mass 0.996; 73 % of segments answered A, 25 % F, 1.4 % `other` | the format is followed but the answers follow the letter, not the content (extreme letter bias); calibrate, and try larger models before the grid |
| [exp02](experiments/exp02_calibration/NOTE.md) | Does contextual calibration (content-free inputs) fix exp01's letter bias? | v3 (superseded, rerun on v4 pending): macro-F1 0.012; `other` F1 0.023 → 0.367; predictions shift to `aa` | no: calibration removes a constant preference but the 0.5B scores hold almost no class information; next, model size |
| [exp03](experiments/exp03_model-size/NOTE.md) | Is the letter bias a matter of model size? exp01's setting with Qwen2.5 1.5B, 3B, 7B (4-bit), raw and calibrated | v3 (superseded, rerun on v4 pending): macro-F1 0.045 (1.5B), 0.107 (3B), 0.206 (7B); calibrated 0.057 / 0.089 / 0.182 | yes: size is the main lever, and the first-letter bias is gone; at k = 0 the 7B still sends short replies (`aa`, `other`) to `bk`; calibration helps only the 1.5B |
| [exp04](experiments/exp04_context-7b/NOTE.md) | Does context (k = 1, 3) fix 7B's short-reply confusion? | pending | pending |
