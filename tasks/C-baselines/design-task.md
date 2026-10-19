# Design task: config-driven baselines in `src/`

## Goal
Design the shared `src/` code so the encoder, decoder and prompting baselines all run through one code path, and every experiment differs only in its `config.yaml`.

## Output
A design document that answers the questions below, with the main interfaces written as function and class signatures. The design is then discussed, and tests are written from it before anything is implemented.

## Constraints
These come from [PROTOCOL.md](../../PROTOCOL.md) and [task B](../B-eval-harness/README.md):
- Experiments contain no code. New behaviour goes behind a config option whose default keeps old configs working (§3.4, §5).
- Runs launch only through `scripts/run.py` and record `config.yaml`, `meta.json`, `history.jsonl`, `metrics.json` and `predictions.jsonl` (§4).
- Predictions follow the harness schema: `seg_id`, `label`, and `scores` for all 7 classes. `null` scores are allowed only for prompting.
- Model selection never touches test. The selected epoch is recorded (§7).
- Labels come from `src/smi/labels.py` only.

---

## Part 1: Data and examples
- What is one "example" that a model sees? Only the target segment, or the target plus surrounding segments?
- If there is context: how many segments, past only or also future, and can it cross meeting boundaries?
- How are speakers represented: dropped, kept raw, or anonymised? Anonymised in what way?
- Is selecting the context (which segments) the same step as formatting it (how they look as text), or separate?
- Is this layer shared by all three families, or does each family build its own?

## Part 2: Turning examples into model input
- What text format does each family need? Is there one format, or one per family?
- How is the target segment marked when context is present?
- When the input exceeds the token limit, what gets truncated first?
- Truncation depends on a tokenizer. Which component owns the tokenizer?

## Part 3: The model abstraction
- What interface must every baseline implement so `run.py` doesn't care which family it's running?
- Are encoder and decoder one implementation with different settings, or two implementations?
- Where does prompting fit, given that it has no training step?
- How does a config select the implementation (a registry, import paths, something else)?
- Where are predictions converted to the harness schema?

## Part 4: Training
- Own training loop or a framework? What does each choice cost in control versus code?
- How is class imbalance handled, and which options does the config expose?
- How is the best epoch chosen? Does that use the harness's metric code, or a separate implementation?
- What is logged each step and each epoch into `history.jsonl`, and by which component?
- Where are checkpoints kept? Is only the best one kept?

## Part 5: Decoder specifics
- How does a decoder produce one label per input (which token, which head)?
- Decoders often have no pad token. What changes because of that? Which padding side?
- Full fine-tuning or parameter-efficient methods (LoRA etc.)? Which settings belong in config?
- Quantization and memory: what has to fit on the GPU, and which settings affect that?

## Part 6: Prompting
- Local model, API model, or both? If both, what interface do they share?
- How are prompts and label definitions stored so that changing a prompt is a config change?
- How are labels shown to the model (names, letters, definitions)?
- Where do `scores` come from? Does the backend expose token log-probabilities? If not, what happens?
- What happens when the output can't be parsed? Every `seg_id` must get a label.
- Few-shot or not: where do examples come from, and how are they chosen reproducibly?
- Cost and repeatability: how many calls does one full val run need? Are responses cached?

## Part 7: Config system
- What file format, and how does an experiment config inherit from a base?
- How are typos and unknown keys handled?
- Where do defaults live, and how is it guaranteed that new options don't change old configs?
- How much of the config is shared across families, and how much is family-specific?

## Part 8: Orchestration and evaluation
- What exactly does `run.py` do, and in what order?
- What does `evaluate()` take, and who calls it?
- Seeds and determinism: what is seeded, and where?
- What goes into `meta.json` (commit, data version, hardware…), and who collects it?

## Part 9: Inputs needed before designing
- Which GPU is available, and how much memory?
- Is there API access, and on what budget?
- Is the target system offline or real-time? This decides whether future context is allowed.

---

## Frameworks and APIs to study

| Framework | Parts | What to find out |
|---|---|---|
| **PyTorch**: `nn.Module`, `DataLoader`, losses, optimizers, AMP, randomness | 2, 4, 5, 8 | What a plain training loop involves; padding via `collate_fn`; class weights in the loss; mixed precision; what seeding does and doesn't guarantee |
| **HF `transformers`**: Auto classes, `AutoModelForSequenceClassification` | 3, 5 | Which architectures it supports (encoders and decoders), how the head is attached, how decoders pick the token to classify from |
| **HF tokenizers**: padding and truncation | 2 | Truncation strategies, text pairs, special tokens, `padding_side` |
| **HF `Trainer` / `TrainingArguments`** | 4 | What it handles for you, how to change the loss (`compute_loss`), `compute_metrics`, callbacks, best-model selection |
| **Alternatives to `Trainer`**: plain PyTorch loop, PyTorch Lightning, HF Accelerate | 4 | How much control versus how much code each one means |
| **HF PEFT (LoRA)** | 5 | What a LoRA config contains, how it wraps a classification model, how the head stays trainable |
| **bitsandbytes (4-/8-bit)** | 5 | Memory savings, QLoRA, and what it constrains |
| **HF chat templates and generation** | 6 | How prompts are formatted for instruction-tuned models, and getting log-probabilities out |
| **vLLM** (optional) | 6 | Fast local inference for prompting tens of thousands of segments |
| **LLM provider APIs** | 6 | Whether they return log-probabilities, rate limits, batch endpoints, cost |
| **Config tools**: plain YAML + dataclasses, OmegaConf, Hydra, pydantic | 7 | How each one does inheritance, validation and rejecting unknown keys |
| **HF `datasets`** (optional) | 1, 2 | Whether it is needed at all versus plain lists |

---

## Resources

All links were checked on 2026-10-11. If huggingface.co doesn't open on your network, use the GitHub link next to each Hugging Face doc instead; the docs there are the Markdown source of the same pages.

### PyTorch
"Learn the Basics" series, in order:
- Overview: https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- Datasets and DataLoaders: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- Building a model (`nn.Module`): https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- Autograd: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- The training loop (loss, optimizer, train/eval steps): https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- Saving and loading: https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

Reference pages:
- `torch.utils.data` (samplers, `collate_fn`, weighted sampling): https://docs.pytorch.org/docs/stable/data.html
- `CrossEntropyLoss` (`weight`, `label_smoothing`): https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Optimizers and LR schedulers: https://docs.pytorch.org/docs/stable/optim.html
- Mixed precision (AMP): https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
- Reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
- All tutorials: https://docs.pytorch.org/tutorials/

### Hugging Face
| Topic | huggingface.co | GitHub mirror |
|---|---|---|
| Transformers (overview) | https://huggingface.co/docs/transformers/index | https://github.com/huggingface/transformers/tree/main/docs/source/en |
| Text classification guide | https://huggingface.co/docs/transformers/tasks/sequence_classification | same folder → `tasks/sequence_classification.md` |
| Padding and truncation | https://huggingface.co/docs/transformers/pad_truncation | same folder → `pad_truncation.md` |
| Trainer | https://huggingface.co/docs/transformers/main_classes/trainer | same folder → `main_classes/trainer.md` |
| Chat templates | https://huggingface.co/docs/transformers/chat_templating | same folder → `chat_templating.md` |
| Text generation | https://huggingface.co/docs/transformers/llm_tutorial | same folder → `llm_tutorial.md` |
| PEFT | https://huggingface.co/docs/peft/index | https://github.com/huggingface/peft (docs: `docs/source`) |
| bitsandbytes | https://huggingface.co/docs/bitsandbytes | https://github.com/bitsandbytes-foundation/bitsandbytes (docs: `docs/source`) |
| Accelerate | https://huggingface.co/docs/accelerate | https://github.com/huggingface/accelerate (docs: `docs/source`) |
| Datasets | https://huggingface.co/docs/datasets | https://github.com/huggingface/datasets |
| LLM course (fine-tuning = chapter 3) | https://huggingface.co/learn/llm-course/chapter3/1 | https://github.com/huggingface/course/tree/main/chapters/en/chapter3 |

### Other frameworks
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/ · https://github.com/Lightning-AI/pytorch-lightning
- vLLM: https://docs.vllm.ai · https://github.com/vllm-project/vllm
- OmegaConf: https://omegaconf.readthedocs.io · https://github.com/omry/omegaconf
- Hydra: https://hydra.cc · https://github.com/facebookresearch/hydra
- pydantic: https://docs.pydantic.dev · https://github.com/pydantic/pydantic

### Papers
Models:
- RoBERTa: https://arxiv.org/abs/1907.11692
- DeBERTaV3: https://arxiv.org/abs/2111.09543

Decoders as classifiers:
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- Label Supervised LLaMA Finetuning: https://arxiv.org/abs/2310.01208

Dialogue acts and context:
- Shriberg et al. 2004, The ICSI MRDA Corpus: https://aclanthology.org/W04-2319/
- Kumar et al. 2018, hierarchical encoder + CRF: https://arxiv.org/abs/1709.04250
- Colombo et al. 2020, guided attention for DA prediction: https://arxiv.org/abs/2002.08801

Class imbalance:
- Focal loss: https://arxiv.org/abs/1708.02002
- Class-balanced loss: https://arxiv.org/abs/1901.05555

Prompting:
- Calibrate Before Use (bias in log-probability scoring): https://arxiv.org/abs/2102.09690
- What Makes Good In-Context Examples: https://arxiv.org/abs/2101.06804
- Sensitivity to prompt format: https://arxiv.org/abs/2310.11324
