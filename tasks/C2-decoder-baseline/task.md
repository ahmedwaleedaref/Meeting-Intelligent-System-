# C2 task description: decoder + classification head (Qwen, LoRA)

The task has three parts: **understand**, then **implement**, then **one experiment** that answers a real question. Don't skip the first part: the implementation is small, and getting it right depends on understanding why each piece is there.

## Part 1: Understand (before writing code)

Be able to answer these, in your own words, in a short note in this task folder.

**LoRA**
1. **What exactly is trained?** The original weight matrix `W` is frozen. LoRA adds a trainable update `ΔW = B·A` (`A`: r × d, `B`: d × r, with r ≪ d) next to it, and the layer computes `W·x + (α/r)·B·A·x`. What does "low rank" mean for `ΔW`, and why is that enough for fine-tuning (intrinsic dimensionality)? How is that different from compressing `W` itself (e.g. with an SVD)?
2. **Why does `B` start at zero?** What does the model look like at step 0?
3. **Why does LoRA save memory?** Which tensors get gradients and AdamW states, and which don't?
4. **Choosing r and α:**
   - Is r a fixed famous default (8, 16) or tuned?
   - What does α/r do, and why must r and α be read together?
   - Which matrices get adapters (`target_modules`: attention only vs all linear layers), and why can that matter more than r?
5. **After training:** do the adapters stay in the forward pass? Merging (`W' = W + (α/r)·B·A`, PEFT's `merge_and_unload()`) vs keeping them separate (small files, several adapters on one base model). Why is merging possible at all?
6. **Where LoRA falls short of full fine-tuning,** and whether that's likely to apply to 7-class classification.

**A decoder as a classifier**
7. **Why the head reads the last real token** (causal attention), and what the fixed suffix (`"\nLabel of [T]:"`) is for.
8. **The pad token:** how the classifier finds the last real token (`transformers/modeling_layers.py`: the rightmost token that isn't `config.pad_token_id`), and the three cases:
   - tokenizer without a pad token: the tokenizer raises;
   - model without `pad_token_id`: the model raises;
   - the two set to different ids: **silently wrong**.

   And the Qwen trap: Qwen's tokenizer pads with `<|endoftext|>` (151643), while Instruct/Qwen3's EOS is `<|im_end|>` (151645), so `model.config.pad_token_id = eos_token_id` would be wrong. DESIGN.md Part 5 has the details.
9. **Memory on a T4:**
   - the frozen base in fp16;
   - the trainable weights (LoRA, head) in fp32, and why `GradScaler` needs that;
   - when 4-bit loading (QLoRA) and gradient checkpointing become necessary.

**Reading**
- Hu et al. 2021, *LoRA: Low-Rank Adaptation of Large Language Models*: https://arxiv.org/abs/2106.09685
- Aghajanyan et al. 2020, *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*: https://arxiv.org/abs/2012.13255
- Biderman et al. 2024, *LoRA Learns Less and Forgets Less*: https://arxiv.org/abs/2405.09673
- Dettmers et al. 2023, *QLoRA*: https://arxiv.org/abs/2305.14314
- Li et al. 2023, *Label Supervised LLaMA Finetuning* (a decoder trained as a classifier): https://arxiv.org/abs/2310.01208
- Hugging Face PEFT docs: https://huggingface.co/docs/peft/index

## Part 2: Implement

Mostly in `load_model_and_tokenizer` (`src/smi/models/model.py`), with options whose **defaults keep the encoder path unchanged** (PROTOCOL §5).

| Where | Add | Why |
|---|---|---|
| model loading | `if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token`, then **always** `model.config.pad_token_id = tokenizer.pad_token_id` | the model must know the tokenizer's pad id (Part 1, q. 8) |
| model loading | a `dtype` option: frozen base in fp16 (default stays fp32) | halves the memory of weights that never train |
| model loading | the PEFT LoRA wrapper: `LoraConfig(task_type="SEQ_CLS", r, lora_alpha, lora_dropout, target_modules)` → `get_peft_model`; print the trainable-parameter count | only adapters + head train; check that the head is trainable |
| model loading | keep the trainable weights (LoRA, `score` head) in fp32 | `GradScaler` can't unscale fp16 gradients |
| model loading | optional 4-bit loading (`BitsAndBytesConfig`) and gradient checkpointing (+ `enable_input_require_grads()` under PEFT) | only when the model or k doesn't fit otherwise |
| `requirements.txt` | pin `peft` (and `bitsandbytes` if 4-bit is used) | reproducibility |
| Collator use | `suffix="\nLabel of [T]:"`; for k > 0 runs, check that `n_truncated == 0` | truncation cuts from the end, where the suffix is (DESIGN.md Part 2, measured window lengths) |
| tests | **pooling test:** for a padded batch, the pooled position is each example's last suffix token | catches every pad-token mistake |
| tests | **default-path check:** the encoder still reproduces its earlier predictions byte for byte | the new options must not change existing runs |

`build_optimizer` (it already skips frozen parameters), `train` (fp16 + GradScaler) and `predict` need no changes. If checkpoints are added later (DESIGN.md Part 4), they save only the adapter and the head.

## Part 3: Experiment: "are we handicapping the decoder?"

On **Qwen 0.5B**, with the same data, k, seeds, epochs and epoch selection as the encoder baseline:

| run | trains | answers |
|---|---|---|
| LoRA r = 16 (attention projections) | adapters + head | the decoder baseline |
| **full fine-tuning** | all parameters (~8 GB of weights + gradients + AdamW at 0.5B: fits a T4 at small batches) | how much LoRA costs on this task |
| LoRA r = 64, or all linear layers targeted | more adapter capacity | whether a gap, if any, is a capacity limit |

- **If LoRA ≈ full fine-tuning at 0.5B,** LoRA is a fair stand-in for larger models, where full fine-tuning no longer fits.
- **If not,** report the gap, and qualify every larger-model result with it.
- **Then scale up one size at a time** only if the 0.5B results suggest it's worth it.

**Every decoder result reports:**
- trainable parameters (count and fraction of the model);
- r, α, target modules, LoRA or full fine-tuning;
- the LoRA vs full fine-tuning gap from this experiment.

## Deliverables
- The Part 1 note.
- The code (Part 2), through a reviewed PR.
- The experiment, run through `scripts/trial_run.py` (extended with the decoder settings), and written up in a `NOTE.md`.
- A recommendation on whether LoRA is a fair stand-in for larger models.
