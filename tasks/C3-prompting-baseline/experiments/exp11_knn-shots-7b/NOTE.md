## Question
Do examples chosen for each segment, the train examples most similar to it, beat exp08's single random draw shown to every segment? exp09 found that the draw alone moves macro-F1 by ~0.05, so 0.341 may partly be a lucky draw. KATE (Liu et al. 2021, "What Makes Good In-Context Examples for GPT-3?", arXiv 2101.06804) retrieves, for each test input, the training examples nearest to it in a sentence encoder's embedding space. With GPT-3 it beat random examples on every task tried, most of all when the encoder was fine-tuned on the task itself.

## Change vs baseline
Baseline, data v3: exp08's `c3_exp08_7b_shots2` (2 examples per label, seed 0, label order, the same 14 for every segment; macro-F1 0.341). Qwen2.5-7B-Instruct 4-bit, k = 0, definitions on, `template_v1.txt`. Notebooks `notebooks/kaggle_c3_exp11.ipynb` (all three runs; only random finished) and `notebooks/kaggle_c3_exp11_knn.ipynb` (the two kNN runs again; see Result), dataset `smi-v3-meetings`.
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
Two notebooks, Kaggle 2 × T4, data v3 (val 16,289 segments):
- `notebooks/kaggle_c3_exp11.ipynb`, commit `b992af6`: only **random** finished. The two kNN runs stopped before the 7B was called: `nearest_per_class` indexed the class rows (on the CPU) with `topk`'s positions (on the GPU), and the unit tests had run on the CPU only. The fix: the class rows now live on the pool's device, and the unit test also runs on a GPU when there is one.
- `notebooks/kaggle_c3_exp11_knn.ipynb`, commit `66ccd9b`: **knn_tuned** and **knn_base**, after checking that step on the GPU. The fix touches only `nearest_per_class`, which the random run does not use.

The pool: 48,495 distinct train examples out of 79,039 segments (`cs` 2,124, `co` 528, `aa` 779, `bk` 679, `ar` 254, `cc` 296, `other` 43,835). The fine-tune ran in both notebooks and gave the same encoder: identical training losses and identical val predictions. Its val macro-F1 was 0.420 / 0.519 / 0.522 after epochs 1–3 (5.5 min each). All three 7B runs:
- passed the cache check: 273 shared tokens; cached 0.12 / 0.074 / 0.050 against padded batch 0.099 / 0.12 / 0.067 (random / knn_base / knn_tuned);
- ran at 2.8–3.0 segments/s, ~95 min each (exp08's shared examples ran at ~20);
- had mean letter mass 0.999.

| v3, k = 0 | macro-F1 | cs | co | aa | bk | ar | cc | other | predicted `other` / `bk` / `aa` |
|---|---|---|---|---|---|---|---|---|---|
| 7B, one random draw for all (exp08, baseline) | **0.341** | 0.372 | 0.215 | 0.376 | 0.139 | 0.548 | 0.397 | 0.882 | 13,151 / 168 / 2,131 |
| 7B, a random draw for each segment | 0.325 | 0.260 | 0.118 | 0.334 | 0.336 | 0.545 | 0.358 | 0.805 | 10,372 / 2,350 / 1,683 |
| 7B, the most similar examples, encoder as released (knn_base) | 0.298 | 0.257 | 0.135 | 0.145 | 0.330 | 0.567 | 0.351 | 0.801 | 11,056 / 2,494 / 662 |
| 7B, the most similar examples, fine-tuned encoder (knn_tuned) | 0.287 | 0.266 | 0.130 | 0.212 | 0.288 | 0.572 | 0.252 | 0.775 | 9,495 / 4,086 / 1,023 |
| retriever alone, encoder as released (no 7B) | 0.163 | 0.114 | 0.112 | 0.206 | 0.294 | 0.187 | 0.066 | 0.852 | 12,046 / 710 / 2,861 |
| retriever alone, fine-tuned encoder (no 7B) | 0.319 | 0.420 | 0.378 | 0.228 | 0.276 | 0.253 | 0.356 | 0.872 | 12,177 / 563 / 2,899 |
| fine-tuned encoder alone (C1's recipe, seed 42, last epoch; no 7B) | 0.522 | 0.494 | 0.442 | 0.402 | 0.611 | 0.718 | 0.463 | 0.931 | 14,101 / 865 / 614 |

(per-class F1; macro-F1 over the 6 targets, without `other`; gold `other` 13,634, `bk` 1,029, `aa` 824)

Recall of each gold class by the 7B:

| gold class (segments) | one draw for all | random per segment | knn_base | knn_tuned |
|---|---|---|---|---|
| `cs` (454) | 0.414 | 0.559 | 0.593 | 0.511 |
| `co` (111) | 0.144 | 0.108 | 0.108 | 0.108 |
| `aa` (824) | **0.675** | 0.508 | 0.131 | 0.238 |
| `bk` (1,029) | 0.081 | 0.552 | 0.566 | **0.716** |
| `ar` (149) | 0.456 | 0.584 | 0.698 | 0.678 |
| `cc` (88) | 0.523 | 0.432 | 0.477 | 0.295 |
| `other` (13,634) | **0.866** | 0.709 | 0.725 | 0.658 |

**The retrieved examples are often the segment's own text.** In knn_tuned's file, 6,611 val segments (41 %; knn_base's: 6,557) have a text that also occurs in train, and the closest example of at least one label is that very text. The pool keeps one example per (label, text), so the prompt shows the text once under each label it has in train. The subsets below are split by how many labels show it; each run is scored on the same segments:

| segments (knn_tuned's file) | n | gold | one draw for all | random per segment | knn_base | knn_tuned | encoder |
|---|---|---|---|---|---|---|---|
| no label shows the segment's text | 9,678 | `other` 8,795, `cs` 452, `aa` 114, `co` 103, `bk` 86, `cc` 85, `ar` 43 | 0.869 / 0.222 | 0.752 / 0.183 | 0.723 / 0.171 | 0.737 / 0.142 | 0.918 / 0.393 |
| one label does | 2,079 | `other` 1,996, `bk` 42, `aa` 24, others ≤ 7 | 0.948 / 0.309 | 0.842 / 0.289 | 0.954 / 0.543 | 0.937 / 0.452 | 0.983 / 0.621 |
| several labels do | 4,532 | `other` 2,843, `bk` 901, `aa` 686, `ar` 99, `co` 3 | 0.526 / 0.207 | 0.444 / 0.257 | 0.446 / 0.216 | 0.263 / 0.227 | 0.718 / 0.418 |

(accuracy / macro-F1 over the 6 targets within the subset; outside `other` the supports are small, so read the macro-F1 as a direction)

- **Several labels** (4,532 segments, 160 distinct texts: "yeah ." 1,230, "mm - hmm ." 551, "OK ." 459, "right ." 372, …) hold 686 of the 824 gold `aa` and 901 of the 1,029 gold `bk`. The prompt shows, for example, "Yeah ." as `aa`, as `bk` and as `other`, with nothing saying which is most frequent.
  - The 7B answers with one of those labels for 4,412 of 4,532 segments (knn_tuned) and 4,477 of 4,485 (knn_base).
  - Identical texts get identical prompts, so every copy of a text gets the same answer. Of the 394 texts that occur more than once in val, 20 / 13 get more than one label from the kNN runs (knn_base / knn_tuned), 181 from random per segment, and 4 from one draw for all.

| text (val segments) | gold | one draw for all | random per segment | knn_base | knn_tuned | encoder |
|---|---|---|---|---|---|---|
| "OK ." (459) | `bk` 283, `other` 168 | `other` all (168 correct) | `bk` 361, `other` 78 (244) | `bk` all (283) | `bk` all (283) | `bk` all (283) |
| "Yeah ." (1,230) | `other` 844, `aa` 287, `bk` 97 | `aa` all (287) | `aa` 903, `other` 169, `bk` 154 (349) | `other` all (844) | `bk` 1,172, `aa` 58 (121) | `other` all (844) |
| "Right ." (372) | `other` 232, `aa` 78, `bk` 62 | `other` all (232) | `bk` 293, `other` 53 (82) | `bk` all (62) | `bk` all (62) | `other` all (232) |

- **One label** (2,079, nearly all `other`): the 7B copies that label (1,985 of 2,079 knn_tuned, 2,021 of 2,072 knn_base) and is right 94–95 % of the time.
- **No label** (9,678; longer segments, with nearly all gold `cs`, `co` and `cc`): the closest examples do not help here either. Macro-F1 within the subset is 0.142 (knn_tuned) and 0.171 (knn_base), against 0.183 (random per segment) and 0.222 (one draw for all).
- **The retriever alone** labels each segment with its most similar example's label. When that example is the segment's own text under several labels, the near-tie is not decided by how often each label goes with the text: "OK ." and "Right ." go to `aa` for every copy, under both encoders. So it is a weak kNN.

The random run:
- **Score:** −0.016 against one draw for all, within the prompt-noise spread (0.036, exp07) and the spread between two draws (0.046, exp09).
- **Labels:** it predicts `bk` 2,350 times (baseline 168) and `cs` 1,497 times (558). `bk` F1 rises 0.139 → 0.336; `cs` falls 0.372 → 0.260 and `other` 0.882 → 0.805.
- **Identical texts:** each segment's answer follows its own examples, so identical texts no longer get identical answers (table above).

The encoder alone: C1's recipe on v3 scores 0.522 at seed 42, in line with C1's 0.5225 on v4 (mean of 3 seeds).

## Conclusion
**Choosing each segment's examples by similarity does not help; it lowers the score.** The 7B scores 0.287 with neighbours from the fine-tuned encoder and 0.298 with the encoder as released. Random examples per segment give 0.325, and exp08's one draw for all gives 0.341.
- Against the baseline, the drops (−0.054, −0.043) are beyond the prompt-noise spread (0.036).
- Against random examples per segment (−0.038, −0.028), they are at or within it.
- With a fine-tuned encoder, KATE on SST-2 gained +5.5 accuracy points over random examples. Here similarity does not pay, and the fine-tuned retriever does no better than the plain one.

**Most of the loss is in the short replies, and it comes from choosing the examples per label.**
- 28 % of val segments are texts like "Yeah ." that occur in train under several labels. Per-label retrieval puts the segment's own text in the prompt once under each of those labels.
- The 7B picks one of them, the same one for every copy, whatever the frequency of each label: "Yeah ." becomes `bk` for all 1,230 copies with knn_tuned, 121 of them correct.
- These segments hold most of gold `aa` and `bk`, so `aa` recall falls from 0.675 to 0.131 / 0.238.
- KATE's own setting, the nearest examples overall with the copies kept, would show many copies of the text, roughly in proportion to its labels in train. That was not tested here.

**Similar examples do not help the longer segments either.** These are the segments whose own text is not in the prompt, and they hold nearly all `cs`, `co` and `cc`. Within them, both kNN runs are below random examples. A dialogue act depends on what a segment does; neighbours by embedding mostly share its wording.

**What the fine-tuned encoder knows does not reach the 7B through examples.** The encoder alone scores 0.522. The 7B shown that encoder's nearest examples scores 0.287: below random examples, and below simply labelling each segment with its single most similar example (0.319).

**A new random draw for each segment scores like one draw for all** (0.325 vs 0.341, within noise). It no longer gives identical texts identical answers, which moves predictions from `other` to `bk`.

So exp08's fixed draw (0.341) stays the best prompting setting.

Limits:
- one 7B run per setting; data v3; 2 examples per label only;
- retrieval per label, over a pool with one example per (label, text); KATE's global nearest examples, without that deduplication, were not run;
- the subset macro-F1 values rest on small supports outside `other`.
