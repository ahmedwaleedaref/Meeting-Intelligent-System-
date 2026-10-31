# C3 — Prompting baseline

## Goal
A prompting baseline (an LLM classifying each segment from a prompt, without fine-tuning) on the same data and evaluation as the encoder and decoder, with scores from label log-probabilities where the backend allows it.

Full description: [task.md](task.md). Design: [DESIGN.md](../C-baselines/DESIGN.md), Part 6.

## Status
Experiments done; in review. Deliverables (task.md): the [Part 1 note](part1-note.md); the code in `src/smi/prompting/`, `scripts/trial_prompting.py`, `scripts/calibrate_prompting.py`, `scripts/pride_prompting.py`, `scripts/retrieve_shots.py`, `configs/prompts/` and `tests/prompting/` (53 tests); experiments exp01–exp11 below, each with a NOTE (exp11 pending). Next: review, then a PR.

Data: release **v4** (`data/v4_meetings`, val 16,188 segments), the same data as the encoder (C1). exp01–exp04 were first run on v3 while v4 was not available locally; they were rerun on v4 with the same code (scores within 0.002 of v3), and every result below is on v4 except exp08–exp11, which ran on v3 (exp08 against its own v3 control). All runs are trial runs of the temporary driver (PROTOCOL §4.1), on val; test was never read.

### Results (val macro-F1 over the 6 target classes)

| Setting (zero-shot, k = 0, definitions on, unless stated) | macro-F1 |
|---|---|
| Qwen2.5-0.5B-Instruct (exp01) | 0.019 |
| 1.5B / 3B (exp03) | 0.046 / 0.107 |
| 7B, 4-bit (exp03) | 0.207 |
| 7B, context k = 1 / k = 3 (exp04) | 0.208 / 0.185 |
| 7B, names only, no definitions (exp06) | 0.217 |
| 7B, two reworded instructions (exp07) | 0.219 / 0.242 |
| 7B, one train example per class (exp05) | 0.298 |
| 7B, one example per class, label / reversed / shuffled order (exp08, v3) | 0.271 / 0.214 / 0.278 |
| 7B, three train examples per class, two draws: seed 0 / seed 1 (exp09, v3) | 0.251 / 0.297 |
| 7B, options rotated (7 orders): range / averaged over the orders / PriDe (exp10, v3) | 0.184–0.316 / 0.297 / 0.186 |
| 7B, two examples per class, options rotated: range / averaged / PriDe (exp10, v3; averaging = 7 × the inference) | 0.223–0.341 / 0.344 / 0.327 |
| **7B, two train examples per class (exp08, v3)** | **0.341** |
| *Encoder, DeBERTa-v3-base fine-tuned (C1 exp01, 3 seeds)* | *0.499* |

- **Model size is the first lever** (0.019 → 0.207 from 0.5B to 7B). Small models answer by letter, not by content.
- **Prompt noise is ~0.04** (exp07): context and definitions off stay within it; **few-shot examples are the change clearly beyond it**: +0.091 with one per class (exp05), +0.070 more with two (exp08). But which examples are drawn matters too: two draws of three per class differ by 0.046, and both score below two per class (exp09).
- **The order of the examples matters** (exp08): with `cs` last instead of `other`, the model over-predicts `cs` and macro-F1 drops by 0.057 (recency bias).
- **The order of the options matters as much as the examples** (exp10): rotating the same 7 options moves zero-shot macro-F1 between 0.184 and 0.316, and with two examples per class between 0.223 and 0.341. Averaging the scores over the 7 orders gives 0.297 zero-shot (+0.091), with no labels and no examples, at 7 × the inference; with two examples per class it gives 0.344, level with the usual order (0.341), which happens to be the best of the seven there, and +0.063 over an order taken at random.
- **Calibration does not help** (exp02, exp03, exp08): the paper's uniform target lowers every 3B / 7B run, the more so the more the model leans to `other` on empty inputs; a train-frequency target is within noise of the raw scores. PriDe's letter prior (exp10) does not help either (0.186 zero-shot, 0.327 with two examples per class, both below the usual order): the prior comes from the probabilities of options the model does not pick, and dividing it out removes correct `other` predictions.
- **The hardest call is the short replies** (`aa` / `bk` / `other`: "yeah", "OK"), the same as for the encoder; each setting moves the confusion between them rather than removing it. At k = 0 identical texts get identical answers, so the examples decide whether all 608 "OK" segments become `bk` or `other` (exp09).
- **Prompting reaches ~68 % of the fine-tuned encoder's score** (0.341 on v3 vs 0.499 on v4) with no training on our data.

### Limits
- One prompting run per setting; the few-shot gains rest on one draw of examples per number of examples (two at three per class, exp09), and the draw alone moves the score by ~0.05, so the best score (0.341) may partly be a favourable draw. The prompt-noise spread was measured on zero-shot prompts only.
- exp08 (the best score), exp09 and exp10 are on v3, the encoder on v4; v3 and v4 differed by ≤ 0.002 wherever both were run.
- The setting was chosen step by step on val (7B, k = 0, few-shot), so the best val score is slightly optimistic; test is untouched.
- The 7B runs are 4-bit, the smaller models fp16, so size and quantization are not fully separated.
- Not done: an API backend and the generation fallback (no API budget); a larger model (~30B, task.md "Open"); the full 16-cell grid (does not fit the GPU budget on the 7B); shifting the `other`-vs-target balance with the saved scores; speaker information in the prompt (not in the shared Dataset).

## Decisions
- Instruct checkpoints, inputs formatted with each model's own chat template (`apply_chat_template`); no instruction tuning or other training of our own — the makers' instruction tuning is what makes the model follow the prompt.
- Scores from the 7 letters' log-probabilities wherever the backend allows it, generation only as the fallback — probabilities allow tuning the `other`-vs-target trade-off on val and give the same evaluation as the fine-tuned models.
- Letters A–G plus descriptive names as answer options, never raw MRDA codes — each answer is one token, so it can be scored by log-probability.
- `other` always defined in the prompt — it's the majority class; without a definition the model over-predicts the targets.
- Unparseable answers fall back to `other`, with the failure rate recorded — `other` is not a target class, so the fallback adds no false positives to the primary metric.
- Few-shot examples stratified (1–3 per class), from train only, with a fixed seed and a fixed order, the same in every cell — every class is shown, PROTOCOL §7, and one change per comparison. Retrieval of similar examples for each segment: exp11.

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
- `--shot-order` (`labels`, `reversed`, `shuffled` with `--shot-order-seed`) reorders the same examples; the default keeps `LABELS` order — in that order the last example is always `other`, and the last example's label pulls the answer (recency bias, exp08).
- One prompt for everything, with the transcript last — instructions are read before the data; definitions on/off swaps only the options block, so the two variants differ in nothing else.
From the results:
- All reported runs on data v4 — the encoder's data, so the numbers are comparable.
- Larger models before any prompt grid (after exp01) — the 0.5B answers by letter, so a grid on it would measure its bias.
- Contextual calibration dropped (exp02, exp03) — it helps only the 1.5B and lowers the 3B and 7B. Rechecked with few-shot prompts in exp08: the uniform target lowers every run; a train-frequency target is within noise of the raw scores.
- Experiments on the 7B, one factor at a time, instead of the 16-cell grid — a 7B val run takes 1.5–2 h, so the grid does not fit the GPU budget.
- k = 0 for exp05–exp07 — k = 1 scores the same (exp04) with shorter, faster prompts.
- A difference counts only if it exceeds the prompt-noise spread (~0.04, exp07).

Execution:
- Each run writes every segment's scores to `scores.jsonl` as it goes; rerunning the same command skips scored segments, and a different setting under the same run name is refused — survives Kaggle's 12 h cap. Local scoring is deterministic, so no cross-run response cache is needed until an API backend exists.
- `--prefix-cache` (off by default): the tokens every prompt of a run shares (instruction, options, examples) go through the model once, and each batch runs only its own tokens — few-shot prompts are mostly shared text. Before scoring, the 8 longest prompts are checked against each full prompt scored alone: the cache must stay within 3 × the usual padded batch's own difference + 0.05, or the run stops (fp16 moves scores a little whenever batching changes: up to 0.08 on the 7B; the + 0.05 was 0.01 until exp10, where a check whose padded batch needed no padding stopped a sound run); the shared length and the check go into `metrics.json`.
- `scripts/calibrate_prompting.py` takes several runs of one model (one model load) and `--target uniform` (the paper, `_cc`) and/or `train-prior` (the train label frequencies, `_ccp`).
- `--option-shift 0..6` rotates the options (PriDe, Zheng et al. 2024): the letters stay A–G, the option texts move up by the shift, and the examples answer with their option's new letter; `scores.jsonl` and the predictions stay in `LABELS` order, so a rotated run is graded like any other. `scripts/pride_prompting.py` takes the 7 rotations of one setting and writes the cyclic average (`_cyclic`) and PriDe (`_pride<d>`: the letter prior from a random 5 % of the segments, no labels, divided out of the rest; one folder per draw), without the model.
- `--shots-file <folder>` gives each segment its own examples instead of `--shots-per-class`'s one set for all (KATE, Liu et al. 2021). `scripts/retrieve_shots.py` writes the folder:
  - `--select similar`: per label, the train examples closest to the segment, by cosine between encoder vectors. With `--fine-tune`, the encoder is first fine-tuned with C1's code and adopted setting, keeping the last epoch.
  - `--select random`: per label, random examples for each segment, as a control.
  - Examples come one per (label, transcript), in `LABELS` order. The file's sha256 and settings go into `settings.json`.
  - Without the LLM, the script also writes and grades the retriever alone (`nearest/`) and, with `--fine-tune`, the encoder's own predictions (`encoder/`).

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
| [exp08](experiments/exp08_calibration-order-7b/NOTE.md) | With examples in the prompt (exp05), does calibration (uniform or train-frequency target) help, and do the order and number of examples matter? (data v3, with its own v3 control) | v3: macro-F1 0.271 (control) / 0.214 (`cs` last) / 0.278 (shuffled) / **0.341** (2 per class); uniform calibration 0.174–0.284, train-frequency 0.227–0.335 | two examples per class is the best prompt (+0.070); order matters through the last example (−0.057 with `cs` last); calibration does not help in either form; the prefix cache cut a run from ~2 h to ~15 min |
| [exp09](experiments/exp09_three-shots-7b/NOTE.md) | Do 3 examples per class beat exp08's 2, and how much does the draw of examples move the score (seeds 0 and 1)? (data v3) | v3: macro-F1 0.251 (seed 0) / 0.297 (seed 1) vs 0.341 with 2 per class; seed 0 predicts `cs` 7,558 times (gold 454), and "OK" goes all to `bk` in seed 0 and all to `other` in seed 1 | no gain from a third example; the draw of examples moves the score beyond noise (0.046 between seeds), so exp08's 0.341 is one draw; choose the examples instead of drawing them at random, or score more draws |
| [exp10](experiments/exp10_pride-7b/NOTE.md) | Does the 7B prefer answer letters whatever they mean, and does removing that letter prior (PriDe, Zheng et al. 2024) help, zero-shot and with 2 examples per class? (data v3) | v3: zero-shot macro-F1 0.184–0.316 over the 7 orders (0.206 in the usual one), averaged over the orders 0.297, PriDe 0.186 (5 draws), letter prior G 0.603; 2 per class 0.223–0.341 (0.341 in the usual one), averaged 0.344, PriDe 0.327, letter prior A 0.488 | the option order moves the score far beyond noise; averaging over the orders removes it (+0.091 zero-shot; level with the usual order with 2 per class, where that order happens to be the best); PriDe does not help in either setting: its prior comes from the probabilities of options the model does not pick |
| [exp11](experiments/exp11_knn-shots-7b/NOTE.md) | Do each segment's own examples, the 2 most similar train examples of every label (KATE, Liu et al. 2021), beat exp08's one random draw for all? Similar by C1's fine-tuned encoder, by the encoder as released, or random per segment (control) (data v3) | pending | pending |
