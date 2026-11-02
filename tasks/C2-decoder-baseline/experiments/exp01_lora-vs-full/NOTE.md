# exp01: LoRA vs full fine-tuning, Qwen2.5-0.5B

## Question
Are we handicapping the decoder? How much does LoRA cost against full fine-tuning on this task, and if there is a gap, is it a capacity limit? This also gives the decoder baseline, to compare with the encoder.

## Setup (change vs the encoder baseline, C1 exp03)
- **Same as C1 exp03:** data v4, k = 0, markers `[T] … [/T]`, 3 epochs, batch 32, warmup 10 %, weight decay 0.01, clipping 1.0, plain cross-entropy, fp16 + fused AdamW, best epoch by val macro-F1, seeds 42, 43, 44 (so the same batches per seed).
- **Changed:** the model (`Qwen/Qwen2.5-0.5B`, the base checkpoint) and what comes with it: the suffix `"\nLabel of [T]:"`, the pad id, and the learning rate. `full` also uses gradient checkpointing (see the memory check).
- **Fixed for all arms:** the suffix wording, and LoRA α = 32, dropout 0.05.

| arm | trains | trainable | base dtype | lr |
|---|---|---|---|---|
| `lora_r16` | adapters on q, k, v, o + head | 2,168,960 (0.439 %) | fp16 | 2e-4 |
| `lora_all_linear` | adapters on all linear layers, r = 16 + head | 8,804,480 (1.782 %) | fp16 | 2e-4 |
| `full` | everything | 494,039,040 (100 %) | fp32 | 1e-5 |

- **Why all-linear rather than r = 64:** both train about the same number of parameters, but all-linear keeps r and α, so it changes only where the capacity sits. r = 64 would also change the step size.
- **Learning rates are chosen, not searched.** LoRA 2e-4 is the top of DESIGN.md's range, full fine-tuning 1e-5 is the encoder's adopted value. A search was left out to keep the run count down. A gap could therefore be an LR mismatch rather than LoRA's limit; if it's large, the conclusion must say so.
- **Differences between the arms beyond the method:** the LoRA arms have an fp16 residual stream, the full arm an fp32 one.

## How it was run
- **First:** `notebooks/kaggle_c2_default_path_check.ipynb` (encoder byte-for-byte check), 1 epoch of DeBERTa at lr 1e-5, seed 42, on `d2f9369` and on `8a0d439`, one per T4: `predictions.jsonl` identical byte for byte (4,382,038 bytes, SHA-256 `8e39edc0…`), val macro-F1 0.4242 on both.
- **Kaggle's environment:** it ships `torchao` 0.10, and `peft` raises on anything below 0.16 as soon as it finds it, so the notebook uninstalls `torchao` (nothing here uses it).
- **Notebook:** `notebooks/kaggle_c2_exp01_lora_vs_full.ipynb` (Kaggle, GPU T4 × 2, two runs at a time). Branch `C2/decoder-baseline`, commit `8a0d439`; the commit and library versions of every run are in its `settings.json`. The notebook file has this run's outputs.
- **Memory check** (longest training batch, 32 × 151 tokens, one T4 of 15.6 GB):
  - `lora_r16`: peak 7.5 GB allocated, 8.0 GB reserved; fits.
  - `lora_all_linear`: peak 10.2 GB allocated, 10.5 GB reserved; fits.
  - `full`: **out of memory at batch 32** (13.7 GiB allocated + 0.7 GiB reserved of 14.56 GiB), so it runs with gradient checkpointing: the same batches and gradients, more time per step. With checkpointing: peak 8.0 GB allocated, 9.3 GB reserved.
- **Command per run:** `CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src python scripts/trial_run.py --mode flat --seeds <seed> --lr <lr> --epochs 3 --name exp01_<arm> --family decoder <arm flags> --data-root <DATA_ROOT> --model Qwen/Qwen2.5-0.5B --out-dir /kaggle/working/runs`, where the arm flags are none (`lora_r16`), `--lora-targets all-linear` (`lora_all_linear`), `--full-finetuning --grad-checkpointing` (`full`).
- **Run folders:** `exp01_<arm>_seed<seed>`.

## Decision rule
Written before the runs. The encoder reference is C1 exp03: 0.5225 ± 0.0059.
- **A gap counts** if the mean difference over the seeds is above both 0.01 and twice its standard error (from the two arms' own spreads), and has the same sign for every seed. The seeds' batches are the same in every arm, so they are compared in pairs. C1's ~0.01 is a working threshold: the decoder's own spread is measured here.
- **`lora_r16` within the threshold of `full`:** LoRA r16 on attention stays the decoder default and is a fair stand-in for larger models. Record the gap with its uncertainty.
- **`lora_r16` behind, `lora_all_linear` within the threshold:** the gap is a capacity limit. All-linear becomes the decoder default, fixed across sizes.
- **Both behind:** report the gap and qualify every larger-model result with it. Check the per-epoch curves and the LR before scaling up.
- **Fewer than 3 seeds finished:** report what ran as an indication, without calling a gap.

## Result
All 9 runs finished with exit code 0, in about 2.5 h on the two T4s. Each one's `settings.json` matches its arm. 16,188 predictions per run, the same seg_ids in all of them, 0 truncated inputs, and the selected epoch's score in `history.jsonl` is bit-identical to `macro_f1_6` in `metrics.json`. Commit `8a0d439`; on Kaggle torch 2.11.0+cu128 (not the pinned 2.14.1), transformers 5.20.1, peft 0.21.2.

Val macro-F1 (6), selected epoch in brackets:

| arm | seed 42 | seed 43 | seed 44 | mean ± std | macro-F1 (7) mean ± std | min per epoch | min per run | peak reserved |
|---|---|---|---|---|---|---|---|---|
| `lora_r16` | 0.4924 (3) | 0.4756 (3) | 0.4741 (3) | **0.4807 ± 0.0102** | 0.5450 ± 0.0088 | 7.1 | 21 | 8.0 GB |
| `lora_all_linear` | 0.4863 (3) | 0.4644 (3) | 0.4873 (3) | **0.4793 ± 0.0130** | 0.5437 ± 0.0111 | 9.5 | 29 | 10.5 GB |
| `full` | 0.4617 (3) | 0.4857 (2) | 0.4709 (3) | **0.4728 ± 0.0122** | 0.5378 ± 0.0105 | 13.8 | 41 | 9.3 GB (checkpointing) |
| *encoder, C1 exp03* | *0.5250* | *0.5267* | *0.5157* | *0.5225 ± 0.0059* | *0.5807 ± 0.0052* | *5.5* | *17* | |

Differences, seed by seed. "2 SE" is twice the standard error from the two arms' own spreads:

| comparison | seed 42 | seed 43 | seed 44 | mean | 2 SE | counts under the rule? |
|---|---|---|---|---|---|---|
| `lora_r16` − `full` | +0.0308 | −0.0102 | +0.0031 | **+0.0079** | 0.0183 | no: below 0.01 and 2 SE, the sign flips |
| `lora_all_linear` − `full` | +0.0246 | −0.0214 | +0.0164 | **+0.0065** | 0.0205 | no |
| `lora_all_linear` − `lora_r16` | −0.0061 | −0.0112 | +0.0132 | **−0.0014** | 0.0190 | no |
| `lora_r16` − encoder | −0.0326 | −0.0511 | −0.0416 | **−0.0418** | 0.0136 | yes |
| `lora_all_linear` − encoder | −0.0387 | −0.0623 | −0.0284 | **−0.0432** | 0.0165 | yes |
| `full` − encoder | −0.0633 | −0.0410 | −0.0448 | **−0.0497** | 0.0156 | yes |

Val macro-F1 (6) and train loss per epoch:

| arm | seed 42 | seed 43 | seed 44 | train loss, epoch 1 → 3 |
|---|---|---|---|---|
| `lora_r16` | 0.349, 0.422, 0.492 | 0.394, 0.450, 0.476 | 0.415, 0.456, 0.474 | 0.49–0.72 → 0.30 |
| `lora_all_linear` | 0.356, 0.406, 0.486 | 0.407, 0.461, 0.464 | 0.449, 0.480, 0.487 | 0.47–0.63 → 0.27 |
| `full` | 0.369, 0.412, 0.462 | 0.394, 0.486, 0.463 | 0.425, 0.452, 0.471 | 0.46–0.47 → 0.25 |

Per-class F1, mean (std) over the 3 seeds:

| arm | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| `lora_r16` | 0.456 (0.002) | 0.363 (0.041) | 0.398 (0.003) | 0.608 (0.003) | 0.716 (0.006) | 0.343 (0.016) | 0.931 (0.000) |
| `lora_all_linear` | 0.410 (0.022) | 0.370 (0.077) | 0.397 (0.002) | 0.606 (0.002) | 0.703 (0.009) | 0.389 (0.044) | 0.930 (0.001) |
| `full` | 0.401 (0.023) | 0.338 (0.018) | 0.394 (0.001) | 0.608 (0.006) | 0.705 (0.010) | 0.391 (0.044) | 0.928 (0.001) |
| *encoder, C1 exp03* | *0.502 (0.013)* | *0.429 (0.015)* | *0.403 (0.006)* | *0.612 (0.001)* | *0.713 (0.008)* | *0.476 (0.011)* | *0.930 (0.001)* |

Confusion, mean over the 3 seeds, the cells that tell the story (gold → predicted):

| | `lora_r16` | `lora_all_linear` | `full` | encoder (k = 0) |
|---|---|---|---|---|
| `cs` → `cs` (of 451) | 179 | 156 | 158 | 224 |
| `co` → `co` (of 111) | 31 | 33 | 29 | 41 |
| `aa` → `aa` (of 817) | 280 | 281 | 279 | 288 |
| `aa` → `other` | 513 | 509 | 509 | 502 |
| `bk` → `aa` | 122 | 122 | 118 | 125 |
| `other` → `aa` | 186 | 195 | 198 | 200 |
| any target → `other` | 1,269 | 1,278 | 1,274 | |
| `other` → any target | 644 | 665 | 710 | |

Prediction agreement over the 16,188 segments: 96.8–97.9 % between two seeds of one arm, 96.8–97.8 % between two arms at one seed.

Training behaviour:
- **Loss spike at the start.** Mean loss over the first 50 steps: `lora_r16` 2.3 / 12.4 / 7.8, `lora_all_linear` 1.9 / 8.9 / 5.5, `full` 1.2 / 2.6 / 1.8 (seeds 42 / 43 / 44), against 1.95 for a uniform guess. All are at 0.45–0.69 by step 200. Within each arm the three seeds then end epoch 3 at the same train loss, so the spike leaves no mark there.
- **Loss scale.** It started at 65,536 and was already 256–512 at step 50 in every run: at least 7–8 halvings, so 7–8 skipped steps out of the first 50. No later drop was logged, except one in `full` seed 44 (by step 800). It then doubled every 2,000 clean steps, ending at 2,048–4,096. No logged step had a non-finite `grad_norm`, but only every 50th step is logged. DeBERTa stayed at 32,768–65,536.
- **Gradient norm before clipping.** Median 4.9–6.2 (`lora_r16`), 2.9–3.5 (`lora_all_linear`), 11–13 (`full`); above the clip of 1.0 on 89–99 % of logged steps. Clipping is active almost always, in every arm.

Memory check, longest batch 32 × 151 tokens: see "How it was run". Time per training step: 160 ms (`lora_r16`), 215 ms (`lora_all_linear`), 325 ms (`full`, checkpointing), against DeBERTa's 119 ms.

## Conclusion
- **LoRA doesn't handicap the decoder on this task.** `lora_r16` is within the rule's threshold of `full` (+0.0079, in LoRA's favour, with a sign that flips between seeds). Training 0.44 % of the parameters cost nothing measurable.
- **The gap isn't a capacity limit, because there is no gap.** All-linear adds 4× the trainable parameters and changes nothing (−0.0014). It fits the training data better (loss 0.30 → 0.27), and so does full fine-tuning (0.25), but none of that reaches val.
- **Decision, by the rule: LoRA r = 16 on the attention projections stays the decoder default.** It's also the cheapest arm: half the time of `full`, and the least memory.
- **Recommendation: LoRA is a fair stand-in for larger models, at k = 0.** Three limits on that:
  - **k = 0 may be hiding a difference.** `aa`, `bk`, `ar` and `other` score the same in all three arms and in the encoder, and the `aa` and `bk` confusions barely move. That's the ceiling of a lone "yeah", not of the method: when no arm can use more capacity, they tie regardless. The gap should be checked again at the first k > 0 setting before relying on it there.
  - **The learning rates weren't searched.** One LR per method. The tie is the less worrying outcome (a badly chosen LoRA LR would show up as a loss for LoRA, and it didn't), but `full` at 1e-5 is also untuned, so it isn't necessarily at its best.
  - **Three seeds resolve only a gap of ~0.02 here.** A real gap below that can't be excluded.
- **The decoder is clearly behind the encoder at this size: −0.042, and the best decoder seed (0.4924) is below the worst encoder seed (0.5157).**
  - It's not the adaptation method: `full` is behind by as much (−0.050).
  - It's three classes: `cc` −0.133, `co` −0.066, `cs` −0.046 account for the whole gap (−0.245 of the −0.251 summed over the 6 classes). These are the segments carrying their meaning in their words. The classes limited by k = 0 are level.
  - It's not a fully fair fight: the encoder's LR and epochs were chosen from a grid, the decoder's weren't, and DeBERTa has dropout.
  - But the scores of the two families don't overlap, so tuning alone would have to find more than 0.02 to reach even the encoder's worst seed.
- **The decoder's seed noise is about twice the encoder's: std 0.010–0.013 against 0.006,** and the arms disagree with each other no more than seeds of one arm do (97 %). Differences below ~0.02 can't be read here, which is larger than C1's working threshold of ~0.01. `co` and `cc` carry most of it, with 111 and 88 val segments.
- **The first 50 steps need attention at larger sizes, though they did no harm here.**
  - The loss spike is larger under LoRA. A likely reason, not tested here: the random head trains at LoRA's higher LR, and α/r doubles the adapters' output. It settled during warmup in every run, so no head LR change is needed now. If a larger model doesn't settle, a separate head LR is the first thing to try, as its own experiment.
  - The loss scale fell from 65,536 to 256–512 at once and stayed 16–32× below DeBERTa's. The skipped steps are a negligible 7–8 of 7,398, nearly all at near-zero LR. Qwen's gradients overflow fp16 much sooner than DeBERTa's, and the `full` arm does it too with fp32 weights, which suggests the cause is the model and not the frozen fp16 base. `train()` doesn't count skipped steps; they're inferred from the scale.
- **Cost:** `lora_r16` is 1.3× DeBERTa's time per epoch, `lora_all_linear` 1.7×, `full` 2.5× with checkpointing. Full fine-tuning didn't fit at batch 32 without checkpointing, so 0.5B is at the edge of what full fine-tuning can do on a T4.
- **Scaling up isn't justified by this result alone.** task.md scales up only if the 0.5B results suggest it's worth it, and they say the decoder is clearly behind, with more trainable capacity not helping. A larger model would be a different question (more pretrained knowledge, not more adaptation), and it remains the team's call: see the task README.
