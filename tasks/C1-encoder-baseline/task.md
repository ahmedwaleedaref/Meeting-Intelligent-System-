# C1 task description: encoder baseline and hierarchical variant

## What already exists
- **Pipeline:** `src/smi/inputs/` (Dataset, length-bucket Sampler, Collator, DataLoader) and `src/smi/models/model.py` (load, predict, AdamW, schedule, train with `fp16` and `fused` options). See [input-pipeline.md](../C-baselines/input-pipeline.md) and [model-and-training.md](../C-baselines/model-and-training.md).
- **Hierarchical model:** `src/smi/models/hierarchical.py`. Stage 1 is `other` vs `not_other`; stage 2 is the 6 targets; `combine` merges them. Each seed writes per-stage reports.
- **Trial runs** (REPORT.md §5): flat seed 42 = 0.503 (still rising at epoch 5); hierarchical seed 42 = 0.488, with stage 1 as the bottleneck (it misses 42% of target segments; ceiling with a perfect stage 2: 0.68).
- **Speed** ([concepts/performance-analysis-and-optimization/05](../../concepts/performance-analysis-and-optimization/05-measured-on-t4.md)): on one T4 the step is now GPU-bound.
- **Notebooks:** `notebooks/kaggle_flat.ipynb` and `notebooks/kaggle_hierarchical.ipynb`, running two independent seeds in parallel, one per T4.

## How runs are done
Through `scripts/trial_run.py` (`--mode flat|hierarchical`, `--seeds`, `--fp32`), or the two Kaggle notebooks. Each seed writes a folder with `settings.json`, one `history.jsonl` per stage, `predictions.jsonl` and `metrics.json`. Settings live in its `SETTINGS` dictionary, plus a few command-line overrides; add an override when an experiment needs to change a setting.

## Remaining work: experiments, roughly in this order

1. **Baseline, 3 seeds:** fp16 + fused, batch 32, k = 0, 5 epochs. The reference every later change is compared against.
2. **Throughput setup:** batch 32 vs 64 (val accuracy, and the LR if needed), and two independent runs in parallel vs one run per session (does the shared CPU slow either down?). Decides the default batch size and how to use the second GPU.
3. **Separator:** `"\n"` vs `tokenizer.sep_token`. DeBERTa turns `"\n"` into a space, so the model sees no segment boundary. **Must be settled before any k > 0 experiment.**
4. **Context:** k ∈ {0, 1, 3, 5}, symmetric and past-only. The largest expected gain: the hierarchical analysis points to context as what separates target segments from `other`.
5. **Training length and LR:** the trial score was still rising at epoch 5. More epochs, and an LR sweep (e.g. 1e-5 / 2e-5 / 3e-5 / 5e-5).
6. **Minority classes:** class-weighted cross-entropy, weights ∝ (1/n_c)^α, α ∈ {0.5, 1}, computed from train counts only. Compare against experiment 1's plain CE. If it's combined with longer training, run one change at a time.
7. **Markers:** plain `[T]` vs single punctuation vs new special tokens, and with vs without markers at k = 0.
8. **Hierarchical model:**
   - 3 seeds;
   - then with context (stage 1 is the bottleneck, and context is its most likely fix);
   - optionally stage 1 with class weights or a tuned threshold;
   - report all three metric files per seed (stage 1, stage 2, combined).
9. **Test evaluation:** only the final configurations, with the project lead's approval (PROTOCOL §7). Needs saved `best` weights, which aren't built yet (DESIGN.md Part 4, checkpoints).

## Per experiment
- reserve the number in this task's README table;
- create `experiments/expNN_<slug>/NOTE.md` with the question, the change, the exact `trial_run.py` command (or notebook) and settings used, the commit, the run folders, the result and the conclusion;
- any new behaviour goes into `src/` behind an option whose default keeps old runs reproducible;
- check the default path byte for byte after adding an option;
- write up negative results the same way as positive ones.
