# 06 How to measure a training step

The method used for [05](05-measured-on-t4.md), and the mistakes it avoids.

1. **Change one thing at a time.** Five changes measured together gave "no speed-up" and no way to tell which change helped and which cancelled it. Each change needs its own before/after.
2. **Predict before you measure.** Write the expected number down first, e.g. "fused AdamW: ~16–20 ms". A result you predicted tests your understanding; a result you only explain afterwards does not.
3. **Synchronise around GPU timings.** `torch.cuda.synchronize()` before each timestamp, or `torch.cuda.Event`. Otherwise you measure how long it took to *queue* the work ([01](01-cpu-gpu-execution.md)).
4. **Warm up, then discard.** The first steps pay one-time costs: kernel selection for new shapes, memory allocator growth. A full warm-up pass over every configuration, thrown away, then the timed steps. (Without it, batch 32 measured first came out 2× too slow.)
5. **Two timing modes.**
   - **total:** no extra syncs, the honest time per step;
   - **breakdown:** a sync between phases, the time of each phase. The syncs remove CPU/GPU overlap, so the breakdown total is a bit higher. The difference is the overlap.
6. **Compare per token, not only per batch.** Different batch sizes give different batches with different padded lengths, so compare the time growth with the **token** growth.
7. **Compare variants in the same session when possible.** Same GPU, same batches. Separate Kaggle sessions can land on different machines or temperatures.
8. **Check your measurement code for side effects.** `scaler.get_scale()` looks harmless but is a `.item()`, a sync. The measurement must not add the very cost it measures.
9. **Measure capacity too.** Peak VRAM (`torch.cuda.max_memory_allocated()`) decides which batch sizes are possible at all.
10. **After adding an option, check that the default path reproduces the old result exactly.** Same seed, same data: the prediction files must be identical byte for byte (GPU runs here were deterministic run to run, so any difference is a real change). This caught a real bug: `torch.optim.AdamW(..., fused=False)` is **not** the default. An explicit `False` also stops PyTorch from switching on its default "foreach" implementation, so it silently falls back to the slow single-tensor loop, with slightly different arithmetic. `fused=None` keeps the default.
11. **Speed is not the only criterion.** A larger batch or fp16 changes the training itself (fewer updates per epoch, different numerics), so its accuracy must be checked on validation before adopting it.
