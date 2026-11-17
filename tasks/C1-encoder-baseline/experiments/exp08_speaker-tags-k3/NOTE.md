# exp08: speaker tags, k = 3

## Question
Does knowing who speaks help? exp07 found that a plain boundary between segments (`[SEP]`) makes no difference. This experiment tags every segment with its speaker instead. How does val macro-F1 (6) compare with exp04 (no tags, 0.5524 ± 0.0103)?

**Why it could help:** whether the previous segment has the same speaker as the target depends strongly on the class (val; train is almost the same):

| | aa | bk | ar | cs | co | cc | other |
|---|---|---|---|---|---|---|---|
| previous segment same speaker | 0.19 | 0.24 | 0.29 | 0.55 | 0.50 | 0.72 | 0.51 |
| next segment same speaker | 0.40 | 0.43 | 0.63 | 0.42 | 0.30 | 0.57 | 0.49 |

- `aa`, `bk` and `ar` are replies: 71–81 % follow a different speaker.
- `cc` mostly continues the same speaker's turn.
- A k = 3 window has 2–4 different speakers in 88 % of the val items (1: 10 %, 5 or more: 2 %).

## Setup (change vs exp04)
- **Same as exp04:** DeBERTa-v3-base, k = 3, separator `"\n"`, lr 1e-5, 3 epochs, batch 32, warmup 10 %, weight decay 0.01, clipping 1.0, plain cross-entropy, fp16 + fused AdamW, no gradient checkpointing, best epoch by val macro-F1, `[T] … [/T]` markers, context clipped at meeting boundaries.
- **The same batches as exp04:** the sampler sorts by the length of the texts without the tags. The first batches' segment ids are identical (checked locally).
- **Changed:** speaker tags (`--speaker-tags`). Each segment starts with `<letter>: `.
  - The target's speaker is always `A`.
  - The other speakers get `B`, `C`, … in order of first appearance in the window, in speaking order.
  - The speaker ids (`mn015`, …) never reach the model. The same people are in train and val, so with the ids it could learn individual people's habits, which wouldn't carry over to new meetings.
  - Example (val):
    ```
    A: Oh , OK
    A: sorry .
    B: OK .
    A: [T] Getting ahead of myself . [/T]
    B: So .
    B: Um .
    B: For those who don't know
    ```
- **How the tokenizer handles it** (checked locally): a tag is plain text, 2 tokens (`▁A`, `:`). There are no new tokens, so there are no new randomly initialised embeddings.
- **Code:**
  - `SegmentItem` has 3 new fields: `left_speakers`, `target_speaker` and `right_speakers`.
  - `SegmentCollator` has `speaker_tags=False`; `build_dataloader` passes it through.
  - `scripts/trial_run.py` has `--speaker-tags`.
  - Without the flag the inputs are byte for byte what they were (checked on all 78 903 train and 16 188 val items).
- **Seeds:** 42, 43, 44, compared seed by seed with exp04 and exp07.
- **Input size, measured locally with the real tokenizer** (batch 32, seed-42 order):

  | | padded length per batch: mean / p99 / max | total tokens per epoch | truncated |
  |---|---|---|---|
  | no tags (exp04) | 81 / 215 / 264 (val max 270) | 6.36 M | 0 |
  | speaker tags | 95 / 229 / 278 (val max 284) | 7.47 M (+17.5 %) | 0 |

- **Memory:** the longest training batch is 32 × 278. exp04's 32 × 264 and exp07's 32 × 270 ran without problems; exp05's 32 × 327 didn't.
  - The notebook checks it with one real step first.
  - **Rule, decided before the run:** plain batch 32. If a run still crashes out of memory (fragmentation, which the check can't see), rerun with `--grad-checkpointing`: the same gradients, ~1/3 more time.
- **Expected time:** ~10–15 % more than exp04, ~34–36 min per run.

## How it was run
- **Notebook:** `notebooks/kaggle_exp08_speaker_tags_k3.ipynb` (Kaggle, GPU T4 × 2, two runs at a time). Branch `C1/encoder-baseline`, commit: _fill in from the `git log` line of the "Code and packages" cell_.
- **Memory check:** _peak allocated / reserved, from section 5_.
- **Command per seed:** `CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src python scripts/trial_run.py --mode flat --seeds <seed> --lr 1e-5 --epochs 3 --past-k 3 --future-k 3 --speaker-tags --name exp08_k3_spk --data-root <DATA_ROOT> --model microsoft/deberta-v3-base --out-dir /kaggle/working/runs`
- **Run folders:** `exp08_k3_spk_seed42`, `exp08_k3_spk_seed43`, `exp08_k3_spk_seed44`.

## Decision rule
_Fill in before the runs:_
- what difference from exp04 counts as the tags helping, hurting, or no different;
- which classes are expected to move (by the table above: `aa`, `bk`, `ar`, `cc`), and whether a gain only there counts.

## Result
_Pending._

## Conclusion
_Pending._
