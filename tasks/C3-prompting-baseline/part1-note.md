# C3 Part 1 note: prompting, understood

Answers to the five questions of [task.md](task.md), Part 1, with evidence from this task's runs (data v4, val, 16,188 segments; experiment notes in [experiments/](experiments/)).

## 1. Instruction-tuned models and chat templates

- **A base model only continues text.** Pretraining teaches next-token prediction on web text; asked to "label this utterance", a base model may continue the text instead of answering.
- **Instruction tuning** trains the model further on instruction → answer pairs, so it treats a request as a request. We use the makers' **Instruct** checkpoints (`Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct`) and train nothing ourselves; few-shot examples are pasted into the prompt and change no weights.
- **Each model has its own chat template** with special tokens marking the speakers. We never write it by hand: `tokenizer.apply_chat_template(..., add_generation_prompt=True)` (`src/smi/prompting/backend.py`). Qwen2.5 adds its default system message when none is given.
- **Where the answer starts:** `add_generation_prompt=True` ends the input with the opened assistant turn, so the next token is the first token of the answer.
- **Which token:** `"A"` … `"G"` without a space are single tokens (ids 32–38) and are what the model emits; `" A"` is a different token (362) and `"A."` is two (32, 13). `load_model` checks that every letter is one token. For the prompt below, the model's six most likely next tokens are all letters without a space: `B` 0.431, `G` 0.220, `F` 0.174, `A` 0.072, `C` 0.037, `E` 0.028. Across runs, the mean probability mass on the 7 letters is ≥ 0.996, except `template_p2` (0.970).
- **Thinking mode:** Qwen2.5 has none. A Qwen3 model would need `enable_thinking=False` in `apply_chat_template`.

The full rendered prompt for one val `aa` segment (`Bed006-c0_0071632_0071872`, k = 1, `template_v1.txt` + `names_v1.txt`), Qwen2.5-0.5B-Instruct:
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Below is part of a transcript of a research meeting. Classify the dialogue act of the segment marked [T] ... [/T].

Options:
A) suggestion
B) command
C) accept
D) acknowledgement
E) reject
F) commitment
G) other

Transcript:
I don't know .
[T] Yeah . [/T]
Thanks .

Answer with the letter of one option only.<|im_end|>
<|im_start|>assistant
```
The 0.5B model answers `B` (command) for this accept: the letter bias of point 3.

## 2. Scoring by log-probabilities instead of generating

- **How:** one forward pass on the prompt; at the last position the model scores every token of its vocabulary (~152k for Qwen); keep the 7 letter logits; softmax over those 7 gives the 7 `scores`, and the label is the highest (`score_batch` in `backend.py`). Only the last position's logits are computed (`logits_to_keep=1`).
- **Against generation:** the label is usually the same (greedy generation's first token is the top token, a letter when the format is followed), but scoring always yields a valid label (no parse failures), 7 real probabilities, one pass instead of several decoding steps, and identical numbers for an identical prompt. It needs the logits, so local models (or APIs returning log-probabilities).
- **Why it matters here:** the data is 83.7 % `other`, and the main error of every model so far is the `other`-vs-target trade-off. With scores, that trade-off can be shifted without changing the prompt, and prompting gets the same evaluation as the fine-tuned models.
- **Prompt health:** `letter_mass`, the share of the full next-token probability on the 7 letters, is logged per segment and averaged in `metrics.json`. Segments below 0.9: 0–152 per run, except `template_p2` (1,471), the one prompt the 7B often answered with something else.

## 3. Biases of prompted classifiers

| Bias | What we saw |
|---|---|
| **Letter / position** | 0.5B (exp01): 73 % of segments answered A, the first option, whatever they said; macro-F1 0.019. Each size has its own favourite (1.5B `bk`, 3B `ar`); only the 7B predicts `other` most (exp03). |
| **Majority label** | Our few-shot examples are one per class, so no label is in the majority. |
| **Recency** | Examples are in `LABELS` order, so `other` is always last; with examples, the 7B predicts `other` more (9,010 → 11,080, exp05). Part of that may be recency; it was not separated from the effect of the examples themselves. |
| **Common-token** | Not measured separately. |
| **Format sensitivity** | Three wordings of the same instruction (exp07): macro-F1 0.207 / 0.219 / 0.242, and per class far more (`ar` F1 0.050 / 0.167 / 0.524). Differences below ~0.04 are noise. |
| **Contextual calibration** | Dividing by the probabilities on content-free inputs ("N/A", "", "[MASK]") (exp02, exp03): helps 1.5B only (0.046 → 0.057), lowers 0.5B, 3B and 7B. For larger models the content-free answer is a sensible one (`other` / `bk`) and the other letters get ≈ 0, so dividing by them inflates rare letters (7B: thousands of `other` → `cs`). Dropped. |

## 4. Few-shot examples

- **Sampling:** stratified, one per class, drawn at random from **train only** with a fixed seed (`src/smi/prompting/shots.py`, `pick_shots`, which refuses any folder but `train`); seg_ids recorded in `settings.json`. Random can miss rare classes; stratified shows every class but looks balanced while the data is 84 % `other`; retrieval (nearest train neighbours) is a later experiment.
- **Order:** fixed, `LABELS` order (`other` last), the same in every run.
- **Format:** each example is rendered with the same function as the query (same markers, same context window), followed by `Answer: <letter>`.
- **How many:** one per class (7 examples). The prompt grows from ~297 to ~437 tokens at k = 0 and ~861 at k = 3, which is why few-shot was run at k = 0.
- **Result (exp05):** 0.207 → **0.298**, the only change clearly beyond prompt noise. `other` predictions move towards the true rate and `aa`, `ar`, `co` improve; acknowledgements are now sent to `aa` (`bk` F1 0.219 → 0.071). One draw of examples only: the spread across seeds is not measured.

## 5. Backends

| | Local model (Hugging Face `transformers`) | Hosted API |
|---|---|---|
| Scores | all logits, so the 7 letter probabilities | only if the provider returns log-probabilities (often top-k only) |
| Cost | free on Kaggle (2 × T4, ~30 GPU-h / week) | per token |
| Repeatability | fixed weights; same prompt, same numbers | the model behind a name can change; record model version and date |
| Limits | memory: 7B only in 4-bit on a 15 GB T4; speed 58 (0.5B) to 2.3 (7B few-shot) segments/s | rate limits |
| Data | stays local | sent to the provider |

Decision: local Hugging Face only. There is no API budget (task.md, "Open"), local scoring gives real scores at no cost, and plain `transformers` is fast enough for one forward pass per segment (no generation), so vLLM was not needed.
