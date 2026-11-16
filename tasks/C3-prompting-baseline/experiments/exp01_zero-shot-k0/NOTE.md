## Question
How well does the smallest prompted model do with the simplest prompt: zero-shot, target segment only (k = 0), label definitions on? This is the reference point for the grid.

## Change vs baseline
None: this is the first prompting run.
- Model `Qwen/Qwen2.5-0.5B-Instruct`, fp16 on one Kaggle T4, letter scoring (softmax over the logits of A–G).
- Prompt `template_v1.txt` + `definitions_v1.txt`, no few-shot examples, `past_k = future_k = 0`.
- Data v4, val split (16,188 segments); first run on v3 (16,289 segments).
- Notebooks `notebooks/kaggle_c3_v4.ipynb` (v4) and `notebooks/kaggle_c3_exp01.ipynb` (v3); driver `scripts/trial_prompting.py` (trial runs, not reportable: PROTOCOL §4.1).

## Runs
- v4: `c3_exp01_v4`
- v3: `c3_exp01_zs_k0_defs`

## Result
### v4 (reported)
commit `ec7f1d4` (same `src/` and prompt files as `c0c893b`), notebook `notebooks/kaggle_c3_v4.ipynb`, Kaggle T4, data v4 (val 16,188 segments, the encoder's data); 51 segments/s.

**val macro-F1 (6 targets) = 0.019**, mean letter mass 0.996.

| | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| F1 | 0.069 | 0.032 | 0.007 | 0.000 | 0.000 | 0.006 | 0.021 |

Predicted: A (cs) 11,777, F (cc) 4,079, G (other) 184, B (co) 141, C (aa) 7. Same picture as v3.

### v3 (first runs, superseded; kept for the record)
Commit `fde4a7e` (same code as `7845789`, message edited), Kaggle T4, fp16, batch 32: 16,289 segments in ~4.7 min (58 segments/s).

**val macro-F1 (6 targets) = 0.019.** Mean letter mass 0.996 (min 0.965): the answer format is followed.

| | cs | co | aa | bk | ar | cc | other |
|---|---|---|---|---|---|---|---|
| P | 0.036 | 0.028 | 0.429 | 0.000 | 0.000 | 0.003 | 0.685 |
| R | 0.934 | 0.036 | 0.004 | 0.000 | 0.000 | 0.136 | 0.012 |
| F1 | 0.069 | 0.032 | 0.007 | 0.000 | 0.000 | 0.006 | 0.023 |

Predicted labels: A (cs) 11,860 = 72.8 %, F (cc) 4,045 = 24.8 %, G (other) 235 = 1.4 %, B (co) 142, C (aa) 7, D (bk) 0, E (ar) 0. Gold: 83.7 % `other`.

## Conclusion
The model answers by letter, not by content: almost every segment gets A (the first option) or F, whatever it says (e.g. 10,625 of 13,634 `other` segments → A; 670 of 1,029 `bk` → F). This is the letter / position bias of task.md point 3, at its extreme. The pipeline itself works (format followed, scores valid), but at 0.5B zero-shot the scores carry almost no class information.

Consequences: a grid of prompt variants on this model would mostly measure this bias. Next: (1) contextual calibration on the saved scores, (2) the same setting on larger Qwen2.5 Instruct models, to see whether the bias is a property of model size.
