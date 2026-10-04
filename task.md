# Structured Meeting Intelligence — Task List

Scope: data transformation, evaluation harness, model baselines.
Each task lists the decisions that must be taken before or during it.

---

## A. Data transformation

### A1. Parse raw `.dadb` into one row per dialogue act
Split segments on `|`, assign word timings to each DA, keep the speaker's own side of quotes. Validate against the published counts (Table 5); port the existing validated parser.
- [ ] **Decision:** how to split timings when the word boundaries of DAs inside a segment are unclear.

### A2. Label mapping to the 7 classes
Merge `aap→aa` and `arp→ar`; everything else becomes `other`.
- [ ] **Decision:** rule for the 18 acts carrying two target tags (priority order? drop?).
- [ ] **Decision:** keep `b` inside `other`, or also produce a variant where `b` and `bk` are merged.

### A3. Handle non-target acts (`z`, `x`, empty)
- [ ] **Decision:** drop them from the stream, or keep them as placeholders (`[nonverbal]`) in context. This affects what k means.

### A4. Decide what goes in the file vs what the model may see
- [ ] **Decision:** general tags kept for analysis only, never as input (proposed: yes).
- [ ] **Decision:** disruption markers (`%--`): input feature, analysis only, or drop.
- [ ] **Decision:** use general tags as an auxiliary multitask target? (Optional; decide now so the field exists.)

### A5. Speaker and timing fields
Store speaker ID, start, end and channel.
- [ ] **Decision:** speaker representation exposed to models: raw IDs, relative (`SAME`/`OTHER`), or per-window letters.

### A6. Adjacency-pair links for layer 2
Store `(proposal_act_id → response_act_id)` pairs; mark proposals with no pair explicitly.
- [ ] **Decision:** multi-responder pairs (`-n`): keep all responders as gold, or only the first.
- [ ] **Decision:** continuations (`+`): link to the first segment of the response, or accept any of them.

### A7. Text normalisation
- [ ] **Decision:** keep disfluencies, fillers and fragments as-is (closer to real speech), or clean them.
- [ ] **Decision:** lowercasing.

### A8. Splits and folds
Standard train/val/test (Table 9), plus K grouped folds over train+val by meeting.
- [ ] **Decision:** K (5?).
- [ ] **Decision:** group folds by meeting, or by meeting series (stricter, but Bmr/Bro dominate).

### A9. Context formatter (shared code, not data)
A single function `format(act, k_left, k_right, speaker_mode) → model input`, used by every pipeline.
- [ ] **Decision:** context unit (utterances, proposed) and the k grid, e.g. {0, 1, 3, 5}.
- [ ] **Decision:** whether both left-only and left+right are run.
- [ ] **Decision:** target marker tokens (e.g. `[T] … [/T]`).

### A10. Freeze and version
Release v1 as JSONL plus a schema doc. Any later change becomes v2, never a silent edit.

---

## B. Evaluation harness

### B1. Prediction schema
- Layer 1: `act_id → label, scores`
- Layer 2: `proposal_id → response_id | none`
- [ ] **Decision:** scores required for every model (needed for threshold tuning and calibration), or labels only.

### B2. Layer-1 metrics
Per-class P/R/F1, confusion matrix, collapsed 3-class level (proposal / response / commitment).
- [ ] **Decision:** primary metric: macro-F1 over the 6 target classes, excluding `other` (proposed).
- [ ] **Decision:** report `cs`+`co` pooled as well as separate.

### B3. Layer-2 metrics
Per-proposal link accuracy (including correct "none") and pair-level P/R/F1.
- [ ] **Decision:** evaluate with gold proposals, predicted proposals, or both (proposed: both).

### B4. Uncertainty
Meeting-level bootstrap CIs on every reported number.
- [ ] **Decision:** number of resamples.
- [ ] **Decision:** add paired bootstrap for "model A vs model B" comparisons.

### B5. CV runner
Train/eval over the folds, aggregate results, enforce the protocol: model selection on folds only, test touched once.
- [ ] **Decision:** who is allowed to run on test, and when.

### B6. Results log
Every run records data version, config, seed and metrics in one place.
- [ ] **Decision:** tool (CSV in repo vs W&B / MLflow).
- [ ] **Decision:** seeds per config (≥3 for small classes).

### B7. Error-analysis report
Generated automatically per run: top confusions with example utterances.

---

## C. Model baselines

### C0. Floor baselines (needed before any neural model)
- "Always other"
- Keyword rules per class
- TF-IDF + logistic regression, at k=0 and with context n-grams
- Layer 2: link to the next act by a different speaker (and a "none" variant). This is the critical one.
- [ ] **Decision:** class weighting scheme, kept fixed across all models.

### C1. Encoder baseline
BERT-base, k=0, weighted cross-entropy, default hyperparameters. Then RoBERTa and DeBERTa-v3 under the same setup.
- [ ] **Decision:** base vs large sizes in the comparison.
- [ ] **Decision:** pooling (`[CLS]` vs target-token mean).
- [ ] **Decision:** fixed hyperparameter budget per model (e.g. a small LR grid only) so the comparison is fair.

### C2. Decoder fine-tuning baseline
Smallest Qwen, classification head, LoRA, k=0.
- [ ] **Decision:** model list and sizes (depends on available GPUs).
- [ ] **Decision:** classification head vs generative labels (proposed: head).
- [ ] **Decision:** LoRA rank and target modules, fixed across models.
- [ ] **Decision:** Llama access (license gating) vs Qwen only.

### C3. Prompting baseline
Small instruct model, zero-shot, k=0, tag definitions in the prompt, constrained output. Then few-shot on the same model.
- [ ] **Decision:** which models.
- [ ] **Decision:** output method (label logprobs vs guided decoding).
- [ ] **Decision:** few-shot design: examples per class, random vs retrieved, drawn from training folds only.
- [ ] **Decision:** inference stack (vLLM vs plain HF).

### C4. Layer-2 link model baseline
Pairwise scorer (proposal, candidate) over the next 5 acts plus a "none" option, using the best encoder from C1.
- [ ] **Decision:** window size (5, from the EDA) and whether same-speaker candidates are allowed.

---

## Blocking decisions — settle first
These unblock everything else:
- **A2** label mapping
- **A3** non-target acts
- **A6** adjacency-pair links
- **A9** context formatter
- **B2** primary metric
- **GPU budget** (determines C2)
