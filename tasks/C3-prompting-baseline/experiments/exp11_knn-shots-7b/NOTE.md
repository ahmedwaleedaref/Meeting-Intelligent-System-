## Question
Do examples chosen for each segment, the train examples most similar to it, beat exp08's single random draw shown to every segment? exp09 found that the draw alone moves macro-F1 by ~0.05, so 0.341 may partly be a lucky draw. KATE (Liu et al. 2021, "What Makes Good In-Context Examples for GPT-3?", arXiv 2101.06804) retrieves, for each test input, the training examples nearest to it in a sentence encoder's embedding space. With GPT-3 it beat random examples on every task tried, most of all when the encoder was fine-tuned on the task itself.

## Change vs baseline
Baseline, data v3: exp08's `c3_exp08_7b_shots2` (2 examples per label, seed 0, label order, the same 14 for every segment; macro-F1 0.341). Qwen2.5-7B-Instruct 4-bit, k = 0, definitions on, `template_v1.txt`. Notebook `notebooks/kaggle_c3_exp11.ipynb`, dataset `smi-v3-meetings`.
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
Pending.

## Conclusion
Pending.
