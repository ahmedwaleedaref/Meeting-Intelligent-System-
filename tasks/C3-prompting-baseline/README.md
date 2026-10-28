# C3 — Prompting baseline

## Goal
A prompting baseline (an LLM classifying each segment from a prompt, without fine-tuning) on the same data and evaluation as the encoder and decoder, with scores from label log-probabilities where the backend allows it.

Full description: [task.md](task.md). Design: [DESIGN.md](../C-baselines/DESIGN.md), Part 6.

## Status
Experiments done; in review. Deliverables (task.md): the [Part 1 note](part1-note.md); the code in `src/smi/prompting/`, `scripts/trial_prompting.py`, `scripts/calibrate_prompting.py`, `configs/prompts/` and `tests/prompting/` (21 tests); experiments exp01–exp07 below, each with a NOTE. Next: review, then a PR.

Data: release **v4** (`data/v4_meetings`, val 16,188 segments), the same data as the encoder (C1). exp01–exp04 were first run on v3 while v4 was not available locally; they were rerun on v4 with the same code (scores within 0.002 of v3), and every result below is on v4. All runs are trial runs of the temporary driver (PROTOCOL §4.1), on val; test was never read.

### Results (val macro-F1 over the 6 target classes)

| Setting (zero-shot, k = 0, definitions on, unless stated) | macro-F1 |
|---|---|
| Qwen2.5-0.5B-Instruct (exp01) | 0.019 |
| 1.5B / 3B (exp03) | 0.046 / 0.107 |
| 7B, 4-bit (exp03) | 0.207 |
| 7B, context k = 1 / k = 3 (exp04) | 0.208 / 0.185 |
| 7B, names only, no definitions (exp06) | 0.217 |
| 7B, two reworded instructions (exp07) | 0.219 / 0.242 |
| **7B, one train example per class (exp05)** | **0.298** |
| *Encoder, DeBERTa-v3-base fine-tuned (C1 exp01, 3 seeds)* | *0.499* |

- **Model size is the first lever** (0.019 → 0.207 from 0.5B to 7B). Small models answer by letter, not by content.
- **Prompt noise is ~0.04** (exp07): context, definitions off and calibration stay within it; **few-shot (+0.091) is the only change clearly beyond it**.
- **The hardest call is the short replies** (`aa` / `bk` / `other`: "yeah", "OK"), the same as for the encoder; each setting moves the confusion between them rather than removing it.
- **Prompting reaches ~60 % of the fine-tuned encoder's score** (0.298 vs 0.499) with no training on our data.

### Limits
- One prompting run per setting; the few-shot gain rests on one draw of examples (seed 0), and the noise spread was measured on zero-shot prompts only.
- The setting was chosen step by step on val (7B, k = 0, few-shot), so the best val score is slightly optimistic; test is untouched.
- The 7B runs are 4-bit, the smaller models fp16, so size and quantization are not fully separated.
- Not done: an API backend and the generation fallback (no API budget); a larger model (~30B, task.md "Open"); the full 16-cell grid (does not fit the GPU budget on the 7B); shifting the `other`-vs-target balance with the saved scores; speaker information in the prompt (not in the shared Dataset).

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
- Prompts are files in `configs/prompts/` (`template_v1.txt`, `definitions_v1.txt`, `names_v1.txt`; `template_p1.txt` and `template_p2.txt` are rewordings for the noise check); a changed prompt is a new file, never an edit; the sha256 of the template and options files is recorded in `settings.json` — every run can be traced to the exact text it used.
- Definitions are paraphrased from the MRDA labelling manual (Dhillon et al. 2004, the sections Accept, Reject, Acknowledgment, Command, Suggestion, Commitment) — the model is judged against the annotators' rules, so it gets the same rules.
- `other` is defined to include plain backchannels ("uhhuh" while someone else is talking) — the project maps the MRDA backchannel tag `b` to `other`, and the manual names `b` / `bk` / `aa` as the most confused tags.
- Few-shot examples are drawn per class with Python's `random.Random(seed)`, in `LABELS` order, with the same context window as the question; their seg_ids go into `settings.json` — the same seed gives the same segments at every k.
- One prompt for everything, with the transcript last — instructions are read before the data; definitions on/off swaps only the options block, so the two variants differ in nothing else.
From the results:
- All reported runs on data v4 — the encoder's data, so the numbers are comparable.
- Larger models before any prompt grid (after exp01) — the 0.5B answers by letter, so a grid on it would measure its bias.
- Contextual calibration dropped (exp02, exp03) — it helps only the 1.5B and lowers the 3B and 7B.
- Experiments on the 7B, one factor at a time, instead of the 16-cell grid — a 7B val run takes 1.5–2 h, so the grid does not fit the GPU budget.
- k = 0 for exp05–exp07 — k = 1 scores the same (exp04) with shorter, faster prompts.
- A difference counts only if it exceeds the prompt-noise spread (~0.04, exp07).

Execution:
- Each run writes every segment's scores to `scores.jsonl` as it goes; rerunning the same command skips scored segments, and a different setting under the same run name is refused — survives Kaggle's 12 h cap. Local scoring is deterministic, so no cross-run response cache is needed until an API backend exists.

## Experiments

| Exp | Question | Result | Conclusion |
|-----|----------|--------|------------|
| [exp01](experiments/exp01_zero-shot-k0/NOTE.md) | Zero-shot, k = 0, definitions on, Qwen2.5-0.5B-Instruct: the reference point for the grid | v4: macro-F1 0.019; letter mass 0.996; 73 % of segments answered A, 25 % F | the format is followed but the answers follow the letter, not the content (extreme letter bias); calibrate, and try larger models before the grid |
| [exp02](experiments/exp02_calibration/NOTE.md) | Does contextual calibration (content-free inputs) fix exp01's letter bias? | v4: macro-F1 0.012; `other` F1 0.021 → 0.359; predictions shift to `aa` | no: calibration removes a constant preference but the 0.5B scores hold almost no class information; next, model size |
| [exp03](experiments/exp03_model-size/NOTE.md) | Is the letter bias a matter of model size? exp01's setting with Qwen2.5 1.5B, 3B, 7B (4-bit), raw and calibrated | v4: macro-F1 0.046 / 0.107 / 0.207 (1.5B / 3B / 7B), calibrated 0.057 / 0.089 / 0.182; v3 within 0.002 | yes: size is the main lever, and the first-letter bias is gone; at k = 0 the 7B still sends short replies (`aa`, `other`) to `bk`; calibration helps only the 1.5B |
| [exp04](experiments/exp04_context-7b/NOTE.md) | Does context (k = 1, 3) fix 7B's short-reply confusion? | v4: macro-F1 0.208 (k = 1), 0.185 (k = 3) vs 0.207 (k = 0); `bk` predicted for 10,112 segments at k = 1 | no overall gain: context helps `aa` / `ar` but the model then reads most segments as replies (`other` recall 0.62 → 0.24); shift the `other`-vs-target balance next |
| [exp05](experiments/exp05_few-shot-7b/NOTE.md) | Does one solved example per label help 7B (k = 0)? | v4: macro-F1 0.298 (zero-shot 0.207); `other` predicted 11,080 times (gold 13,543); `aa` / `ar` / `co` F1 up, `bk` down to 0.071 | yes: the only change clearly beyond prompt noise (+0.091); the short-reply confusion moves from `bk` to `aa`; one example draw only |
| [exp06](experiments/exp06_no-definitions-7b/NOTE.md) | Do the label definitions help 7B, or are the names enough? | v4: macro-F1 0.217 vs 0.207 with definitions; without them `aa` F1 0.000, `cc` 0.043, `ar` 0.557 | no difference on average (within noise), but the definitions decide which classes the model uses; names only is 2.8 × faster |
| [exp07](experiments/exp07_prompt-noise-7b/NOTE.md) | How much does rewording the instruction move the 7B score (prompt noise)? | v4: macro-F1 0.207 / 0.219 / 0.242 for three wordings; spread 0.036 | differences below ~0.04 are noise: few-shot is real, definitions off and context are not |
