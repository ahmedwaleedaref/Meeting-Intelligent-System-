# C3 task description: prompting baseline

Three parts again: **understand**, **implement**, **experiment**. The design is in [DESIGN.md](../C-baselines/DESIGN.md), Part 6; this file lists what is left to do.

## Part 1: Understand (before writing code)

Be able to answer these in a short note in this task folder:
1. **Instruction-tuned models and chat templates:**
   - **Why a base model isn't enough.** An LLM is first pretrained on web text to predict the next token. Given an instruction ("label this utterance"), a base model doesn't necessarily follow it: it may just continue the text the way some web page would, e.g. "labelling is for nerds".
   - **Instruction tuning** fixes that. The model is trained further on many examples of an instruction followed by the answer we want, so it learns to treat the request as a request and answer it.
   - **Each model has its own chat template,** with special tokens that mark who is speaking, e.g. Qwen's `<|im_start|>system … <|im_end|>`, `<|im_start|>user …`, `<|im_start|>assistant`. The model was instruction-tuned in that format, so its input must use it.
   - **We don't do instruction tuning ourselves.** The model's makers already did. We use the **Instruct** checkpoint (e.g. `Qwen/Qwen2.5-0.5B-Instruct`, not the base `Qwen/Qwen2.5-0.5B`), and format every input with its own template through `tokenizer.apply_chat_template`; we never write the template by hand. In prompting nothing is trained. Few-shot examples are pasted into the prompt (in-context learning) and don't change the model.
   - **Where the answer starts:** `add_generation_prompt=True` ends the prompt with the opened assistant turn, so the model's next token is the first token of its answer. Which token is that for the 7 options (`"A"` vs `" A"`)? Check the model's top next-token predictions.
   - **Thinking mode:** does the model's template add a reasoning block (Qwen3: `<think> … </think>`), and how is it switched off (Qwen3: `enable_thinking=False`)?
   - **Show, for the chosen model,** the full rendered prompt for one real val segment. A worked example, the prompt for one val `aa` segment with Qwen2.5-0.5B-Instruct:
     ```
     <|im_start|>system
     You label dialogue acts in meeting transcripts.<|im_end|>
     <|im_start|>user
     Meeting excerpt:
     u That 's pretty good , I think .
     I don't know .
     [T] Yeah . [/T]
     Thanks .
     OK .

     Which dialogue act is the utterance marked [T]?
     A) suggestion
     B) command
     C) accept
     D) acknowledgement / backchannel
     E) reject
     F) commitment
     G) none of the above

     Answer with one letter.<|im_end|>
     <|im_start|>assistant
     ```
2. **Scoring by log-probabilities, instead of generating the answer:**
   - **How it works:**
     1. one forward pass on the prompt;
     2. at the last position the model gives a score for **every** token in its vocabulary (151,936 for Qwen);
     3. keep only the 7 letter tokens, and take a softmax over those 7, giving one probability per class.
     4. In the worked example above: `C) accept` = 0.78, the correct label.
   - **Is it different from letting the model generate the answer?** Usually not in the label. Greedy generation's first token is the top token over the whole vocabulary, so when that's a letter, generating and scoring pick the same class. They differ in what else you get:

     | | generate the answer | read the 7 letter log-probabilities |
     |---|---|---|
     | output | a label only (`scores: null`) | 7 probabilities |
     | the model wants to say something else (`The answer is…`, `**C**`) | text to parse: can fail, needs a retry, falls back to `other` | always a valid label; never a parse failure |
     | cost | a forward pass plus a few decoding steps | one forward pass |
     | availability | every backend | needs the logits: local models, or APIs that return log-probabilities |

   - **Why it matters for this project:**
     - **The data is 83% `other`,** and the main error found so far (REPORT.md §5) is target segments predicted as `other`. With 7 probabilities, that trade-off can be adjusted **without changing the prompt**: e.g. predict a target class when its probability passes a threshold tuned on val. With generated labels there is no number to adjust.
     - **The scores also give prompting the same evaluation as the fine-tuned encoder and decoder** (PR-AUC, calibration), not only macro-F1. That makes the three families comparable on equal terms.
     - So: log-probabilities for every backend that allows it; generation only as the fallback for APIs without them.
   - **Which token is the answer:** `"A"`, `" A"` and `"A."` are different tokens (Qwen: 32, 362, and 32 + 13). Read the form the model actually emits after the prompt; check its top next-token predictions. In the worked example the top tokens were letters without a space.
   - **Prompt health:** log the probability mass that falls on the 7 letters out of the full vocabulary (1.000 in the worked example). A low value means the model wanted to answer something else, and its scores are less trustworthy.
3. **Biases of prompted classifiers:** preferences that don't depend on the input.
   - **Letter and position bias:** preferring certain option letters, or the first or last option, whatever they stand for.
   - **Majority-label bias:** preferring the label that appears most often among the few-shot examples.
   - **Recency bias:** preferring the label of the last few-shot example.
   - **Common-token bias:** preferring answers that were frequent in pretraining.
   - **Format sensitivity:** small changes in spacing, separators, casing or wording moving the results.
   - **Contextual calibration:** a correction that measures the model's probabilities on a content-free input (e.g. "N/A") and divides real predictions by them.

   References: Zhao et al. 2021, *Calibrate Before Use* (https://arxiv.org/abs/2102.09690), the core paper on these biases and on calibration; Zheng et al. 2023, *Large Language Models Are Not Robust Multiple Choice Selectors* (https://arxiv.org/abs/2309.03882), on option-letter bias.
4. **Few-shot examples:** a few solved examples (a segment plus its correct answer) placed in the prompt before the segment to label. The model imitates them; nothing is trained. The questions to answer:
   - **Which sampling strategy?** How are the examples drawn from train?
     - **random:** simple, but rare classes may never appear;
     - **stratified** (e.g. 1–2 per class): every class gets shown, but the examples look balanced while the real data is 83% `other`, so the model may over-predict the target classes;
     - **retrieval:** see below.
   - **What order?** The label of the last example pulls the answer (recency bias, point 3).
   - **What format?** The examples must look exactly like the query: with context windows when k > 0, and the same markers and answer format.
   - **How many?** Each example adds its window to every prompt. At 7–14 examples with context, the prompt gets several times longer, and so do time and cost.
   - **From where?** Train only, never val or test (PROTOCOL §7); record the examples' seg_ids with the run.

   **The current design** (DESIGN.md Part 6) is **stratified** (1–2 per class), **seeded** (a fixed random seed), in a **fixed order**, with the **same examples in every cell** of the grid. It's the simplest choice that shows every class and keeps runs comparable.

   **More advanced, as a later experiment: retrieval.** For each query, pick the train segments most similar to it (nearest neighbours by sentence-embedding similarity). It's often stronger, but the examples differ for every query, so there's no shared prompt prefix to cache, and it costs more. Reference: Liu et al. 2021, *What Makes Good In-Context Examples for GPT-3?* (https://arxiv.org/abs/2101.06804).
5. **Backends:** what a local model (Hugging Face or vLLM) gives you, against an API (log-probabilities or not, rate limits, cost, model versions changing behind the same name).

**Reading**
- Zhao et al. 2021, *Calibrate Before Use: Improving Few-Shot Performance of Language Models*: https://arxiv.org/abs/2102.09690
- Liu et al. 2021, *What Makes Good In-Context Examples for GPT-3?*: https://arxiv.org/abs/2101.06804
- Sclar et al. 2023, *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design*: https://arxiv.org/abs/2310.11324
- Hugging Face, *Chat templates*: https://huggingface.co/docs/transformers/chat_templating
- Hugging Face, *Text generation*: https://huggingface.co/docs/transformers/llm_tutorial
- vLLM docs (optional, for fast local inference): https://docs.vllm.ai

## Part 2: Implement

**Reuse, don't rebuild:**
- **`SegmentDataset`** (`src/smi/inputs/dataset.py`): the same items as the other families. The test split needs `allow_test=True` and has no labels.
- **The harness:** `write_predictions` accepts `scores: null` for backends without log-probabilities (all records in a file must then be null). `evaluate()` writes `metrics.json`.

**To write** (in `src/`, through a reviewed PR):
1. **Prompt templates and label definitions as files** in one shared folder:
   - immutable (a changed prompt is a new file);
   - referred to by name in the config;
   - a hash of the rendered template recorded with the run.
2. **The prompt renderer:** item → instruction + definitions (optional) + few-shot examples (optional) + the marked window (`[T] … [/T]`, reversing `left` as the Collator does) + the question, wrapped in the chat template.
3. **The backend interface:** prompt + the 7 label tokens in; text plus label log-probabilities (if available) out. Implementations:
   - a local model (log-probabilities from the logits);
   - optionally a hosted OpenAI-compatible API (`top_logprobs`).
4. **Scores:**
   - from log-probabilities: a softmax over the 7 letters; log the mass outside them;
   - without log-probabilities: `scores: null` and parsing (constrain the output → strict parse → one retry → fallback `other`), with the failure rate in `metrics.json`.
5. **A response cache,** keyed by model + rendered prompt + decoding parameters: reruns are free and repeatable.
6. **A `predict`-style function** that returns records in the same shape as `predict()` in `src/smi/models/model.py` (`{seg_id: {"label", "scores"}}`), so `write_predictions` and `evaluate()` work unchanged; and a mode in `scripts/trial_run.py` (or a sibling script) to run it.

## Part 3: Experiments

**Grid** (DESIGN.md Part 6): definitions on/off × shots 0 / 1–2 per class × k ∈ {0, 1, 3, 5} = 16 cells.
- Run the grid with one fixed local model, preferably the same family as the decoder baseline (C2), so prompting vs fine-tuning is compared on one model.
- **Prompt noise:** 2–3 paraphrases of the template on the best cell. Differences between cells smaller than the paraphrase spread are not real.
- **Scale:** only the best 1–2 configurations go to a larger model (hosted, with log-probabilities), and optionally one frontier model (label only), if the budget allows.
- **Cost:** one call per segment, 16,188 per val run. Estimate the time locally, and the cost for any API, before running the grid.

**Every result reports:**
- the model and its version (and the date, for APIs);
- the template name and hash;
- the shots' seg_ids;
- the parse-failure rate or the mass outside the 7 letters.

## Open, decide first
- **The API budget** (DESIGN.md Part 9): decides whether any hosted or frontier model is used at all.
- **Which larger model, and where to run it?** For the "what does scale buy?" runs, a strong ~30B open model such as Qwen3.5-27B. Questions to answer with measurements, not assumptions:
  - **On Kaggle's 2 × T4** (2 × 15 GB, no bf16): does a 4-bit (GPTQ/AWQ) version fit across both GPUs with vLLM (`--tensor-parallel-size 2`, `--dtype half`), with room for the KV cache? How many seconds per prompt, so how many hours per full val run (16,188 prompts)? Does prefix caching of the shared instruction help enough?
  - **Mixture-of-experts alternatives** (e.g. Qwen3.6-35B-A3B): all weights must fit, but only ~3B parameters are active per token, so they're much faster. Do they fit on 2 × T4?
  - **What's the largest model we can prompt on free Kaggle GPUs** at an acceptable speed (~30 GPU-h per week, 12 h sessions)?
  - **Hosted APIs** (e.g. OpenRouter): what does one val run cost, and **does the provider return log-probabilities** for that model (needed for scores, point 2)?

  **Hint, a possible plan:** run the grid with **small models locally** (Qwen2.5 / 3.5 at 0.5B–4B: free, with log-probabilities), then take only the best configuration to **a ~30B model through an API**, if it returns log-probabilities. That's likely cheaper and much faster than running 27B on T4s (a rough estimate: a few dollars per val run by API, against several hours per run on 2 × T4). Check both before deciding.

  **Reading to answer these questions** (enough to reason about memory and speed, not in depth):
  - **KV cache** (what fills the memory next to the weights, and what prefix caching reuses):
    - Hugging Face blog, *KV Caching Explained*: https://huggingface.co/blog/not-lain/kv-caching
    - Hugging Face docs, *Cache strategies*: https://huggingface.co/docs/transformers/kv_cache
    - vLLM docs, *Automatic Prefix Caching*: https://docs.vllm.ai/en/latest/design/prefix_caching.html
    - Kwon et al. 2023, *PagedAttention* (how vLLM manages KV-cache memory): https://arxiv.org/abs/2309.06180
  - **Mixture-of-experts models** (why "35B total, 3B active" is fast, but still needs memory for all 35B):
    - Hugging Face blog, *Mixture of Experts Explained*: https://huggingface.co/blog/moe
    - Jiang et al. 2024, *Mixtral of Experts*: https://arxiv.org/abs/2401.04088
    - Fedus et al. 2021, *Switch Transformers*: https://arxiv.org/abs/2101.03961
  - **4-bit quantization** (what GPTQ / AWQ do to the weights):
    - Hugging Face docs, *Quantization overview*: https://huggingface.co/docs/transformers/main/en/quantization/overview
    - Frantar et al. 2022, *GPTQ*: https://arxiv.org/abs/2210.17323
    - Lin et al. 2023, *AWQ*: https://arxiv.org/abs/2306.00978

## Deliverables
- The Part 1 note.
- The code (Part 2), through a reviewed PR.
- The grid and the noise check, each written up in a `NOTE.md`.
