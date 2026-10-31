## Question
Do examples chosen for each segment, the train examples most similar to it, beat exp08's single random draw shown to every segment? exp09 found that the draw alone moves macro-F1 by ~0.05, so 0.341 may partly be a lucky draw. KATE (Liu et al. 2021, "What Makes Good In-Context Examples for GPT-3?", arXiv 2101.06804) retrieves, for each test input, the training examples nearest to it in a sentence encoder's embedding space. With GPT-3 it beat random examples on every task tried, most of all when the encoder was fine-tuned on the task itself.

## Change vs baseline
Baseline, data v3: exp08's `c3_exp08_7b_shots2` (2 examples per label, seed 0, label order, the same 14 for every segment; macro-F1 0.341). Qwen2.5-7B-Instruct 4-bit, k = 0, definitions on, `template_v1.txt`. Notebooks `notebooks/kaggle_c3_exp11.ipynb` (all three runs; only random finished) and `notebooks/kaggle_c3_exp11_knn.ipynb` (the two kNN runs again, see Result), dataset `smi-v3-meetings`.
- Each segment gets its own 14 examples (`--shots-file`, made by `scripts/retrieve_shots.py`): 2 train examples of every label, in exp08's order (labels `cs` first, `other` last), the closest first within a label. Examples come from train, one per (label, transcript): at k = 0, thousands of train segments are just "yeah" or "OK", and copies would show the same example twice.
- Three ways of choosing them:
  - **knn_tuned**: cosine between the `[CLS]` vectors of C1's encoder, DeBERTa-v3-base fine-tuned on train with C1's code and adopted setting (`smi.models.model.train`, `scripts/trial_run.py`'s settings, lr 1e-5, 3 epochs, seed 42). The last epoch is kept, so the val labels choose nothing. This is KATE's best variant for classification (an encoder fine-tuned on the task: KATE_sst-2).
  - **knn_base**: cosine between the mean last-layer vectors of DeBERTa-v3-base as released, not fine-tuned (KATE_roberta). Similarity of wording only.
  - **random**: 2 random examples of every label, drawn for each segment (seed 0) from the same pool. This is the control: different examples for each segment, without similarity. KATE's random baseline also draws per test input.
- From the same files, without the 7B:
  - `nearest`: each segment gets the label of its closest example. This is the retriever alone (KATE's kNN baseline): does the LLM add anything?
  - The fine-tuned encoder's own predictions: C1's model on v3, seed 42, last epoch.
- Speed: `--prefix-cache`, batch 16. Only the instruction and options are shared now. A run stopped by its cache check reruns without the cache as `<run>_nocache`.

Differences from the paper:
- KATE takes the k nearest overall, whatever their labels. Taking them per label is our choice: every label stays in the prompt, as in exp08, so only which examples are shown changes.
- KATE's encoders are RoBERTa-large; C1's is DeBERTa-v3-base.
- knn_tuned's prompt carries what an encoder learned from our train labels, so it is prompting plus a fine-tuned retriever, and is compared with that encoder alone. knn_base and random are prompting only.

What the paper leads us to expect:
- Its one classification task (SST-2 examples, IMDB test, 3 examples, GPT-3) scored:
  - random: 87.95 ± 2.74;
  - KATE with plain RoBERTa: 91.99;
  - KATE with RoBERTa fine-tuned on SST-2: 93.43.
- The retriever alone (kNN) scored 50.2 with plain RoBERTa and 92.46 with the fine-tuned one.
- Its largest gains are in QA and table-to-text, where the neighbours carry the answer's facts.
- A dialogue act depends on what a segment does more than on its topic. At k = 0, identical texts ("yeah") still get identical examples and so identical answers.

## Runs
- v3: `c3_exp11_7b_knn_tuned`, `c3_exp11_7b_knn_base`, `c3_exp11_7b_random`
- shots files:
  - `shots/knn_tuned`, with `nearest/`, `encoder/` and `history.jsonl`;
  - `shots/knn_base`, with `nearest/`;
  - `shots/random`.

## Result
First notebook: commit `b992af6`, `notebooks/kaggle_c3_exp11.ipynb`, Kaggle 2 × T4, data v3 (val 16,289 segments). The pool: 48,495 distinct train examples out of 79,039 segments (`cs` 2,124, `co` 528, `aa` 779, `bk` 679, `ar` 254, `cc` 296, `other` 43,835).
- **random** finished. The cache passed its check: 273 shared tokens; cached 0.12, padded batch 0.099, allowed 0.35. Speed 2.8 segments/s (~97 min; exp08's shared examples ran at ~20). Mean letter mass 0.999.
- **knn_tuned and knn_base** stopped before the 7B was called. `nearest_per_class` indexed the class rows (on the CPU) with `topk`'s positions (on the GPU), and the unit tests had run on the CPU only.
  - The fix: the class rows now live on the pool's device, and the unit test also runs on a GPU when there is one.
  - The two runs are made again in `notebooks/kaggle_c3_exp11_knn.ipynb`, which first checks this step on the GPU.
  - The random run does not use `nearest_per_class`, so it stands.
  - The fine-tune itself had finished: val macro-F1 0.420 / 0.519 / 0.522 after epochs 1–3, 5.5 min each. The rerun fine-tunes again; fp16 training on a GPU is not bit-exact, so its encoder may differ slightly.

So far:

| v3, k = 0 | macro-F1 | cs | co | aa | bk | ar | cc | other | predicted `other` / `bk` |
|---|---|---|---|---|---|---|---|---|---|
| 7B, one random draw for all (exp08, baseline) | **0.341** | 0.372 | 0.215 | 0.376 | 0.139 | 0.548 | 0.397 | 0.882 | 13,151 / 168 |
| 7B, a random draw for each segment | 0.325 | 0.260 | 0.118 | 0.334 | 0.336 | 0.545 | 0.358 | 0.805 | 10,372 / 2,350 |
| fine-tuned encoder alone (C1's recipe, seed 42, last epoch; no 7B) | 0.522 | 0.494 | 0.442 | 0.402 | 0.611 | 0.718 | 0.463 | 0.931 | 14,101 / 865 |

(per-class F1; macro-F1 over the 6 targets, without `other`; gold `other` 13,634, `bk` 1,029)

The random run:
- **Score:** −0.016 against the one draw for all, within the prompt-noise spread (0.036, exp07) and the spread between two draws (0.046, exp09).
- **Labels:** it predicts `bk` 2,350 times (baseline 168) and `cs` 1,497 times (558). `bk` F1 rises 0.139 → 0.336; `cs` falls 0.372 → 0.260 and `other` 0.882 → 0.805.
- **Identical texts no longer get identical answers.** 394 texts occur more than once in val (6,394 segments), and 181 of them get more than one label (baseline: 4, from fp16 batching). Each segment's answer follows its own examples:

| text (val segments) | gold | random per segment | baseline | encoder |
|---|---|---|---|---|
| "OK ." (459) | `bk` 283, `other` 168 | `bk` 361, `other` 78 (244 correct) | `other` for all (168 correct) | `bk` for all (283 correct) |
| "Yeah ." (1,230) | `other` 844, `aa` 287, `bk` 97 | `aa` 903, `other` 169, `bk` 154 (349 correct) | `aa` for all (287 correct) | `other` for all (844 correct) |
| "Right ." (372) | `other` 232, `aa` 78, `bk` 62 | `bk` 293, `other` 53 (82 correct) | `other` for all (232 correct) | `other` for all (232 correct) |

- **The encoder alone:** C1's recipe on v3 scores 0.522 at seed 42, in line with C1's 0.5225 on v4 (mean of 3 seeds).

kNN runs: pending (`notebooks/kaggle_c3_exp11_knn.ipynb`).

## Conclusion
Pending the kNN runs.
