# The input pipeline (`src/smi/inputs/`)

How a split folder of meeting files becomes model-ready batches. Four pieces, built in this order, each with one job:

```
per-meeting files ──► SegmentDataset ──► LengthBucketBatchSampler ──► SegmentCollator ──► model
                      what ONE example     which examples go           how ONE batch
                      is (plain text)      together in a batch         becomes tensors
                      └──────── model-independent ─────────┘           └─ model-dependent ─┘
                                        build_dataloader wires them into one PyTorch DataLoader
```

**Why the split matters:**
- **The Dataset and Sampler know nothing about the model:** no tokenizer, no special tokens, no text formatting. So the encoder, the decoder and prompting can all use them.
- **Everything that depends on the model** (the tokenizer, how text is laid out, the decoder's suffix) lives in the Collator, which each model family builds for itself.

The design decisions behind these choices are in [DESIGN.md](DESIGN.md), Parts 1–2. This document describes the logic of the code.

---

## 1. `SegmentDataset` (`dataset.py`): what one training example is

**Job:** load one split and define **one training example**: one target segment, its label, and the text around it. The rest of the pipeline only asks it two questions: "how many examples are there?" (`__len__`) and "give me example `i`" (`__getitem__`).

**One item (`SegmentItem`):**
```python
{
    "seg_id":      "Bed006-cA_0028700_0029130",                       # target id , the join key with the harness
    "label":       6,                                                  # index into LABELS ("other") ; None on test
    "left":        ["Oh , OK", "Some in some introductions are in order ."],   # closest FIRST
    "target_text": "sorry .",
    "right":       ["OK .", "Getting ahead of myself ."],             # closest first
}
```
- **Plain text, not token ids and not one joined string.** How the text is laid out differs per family, so the Dataset leaves it to the Collator or the prompt renderer. That's what keeps it general.
- **`left` is in reverse speaking order** (closest first), so truncation can always drop the last element. Every renderer must reverse it.

**Logic of `__init__(split_dir, past_k, future_k, allow_test=False, keep_labels=None)`:**
1. **Check the arguments:**
   - `past_k` and `future_k` must be ≥ 0;
   - a `test/` folder is refused unless `allow_test=True` (PROTOCOL §7: test labels are read only behind an explicit flag);
   - `keep_labels` is refused on test, which has no labels to filter on.
2. **List the meeting files `*.jsonl` and sort them by name.** Directory order differs between machines; sorting makes example `i` the same segment everywhere, so a seeded shuffle gives the same batches everywhere.
3. **For each meeting file,** read all its rows. They are already in position order: `make_meeting_files.py` writes them that way.
4. **For each row (a possible target):**
   1. **Skip it if `keep_labels` is given and its label isn't in it.** Used by the hierarchical model's stage 2, which trains only on target classes. Context still comes from *every* row of the meeting, filtered or not.
   2. **Build `left`:** step back 1, 2, … `past_k` rows; stop at the start of the meeting.
   3. **Build `right`:** step forward 1, 2, … `future_k` rows; stop at the end of the meeting.
   4. **The label:** `None` on test; otherwise its index in `LABELS`, raising an error if it isn't one of the 7 classes.
   5. **Store the item.**
5. **Keep all items in a list** (`self.data`).

**`__len__`** returns the number of items. **`__getitem__(i)`** returns item `i` as it is, not a copy.

**Rule for every later piece:** items are **read-only**. They are stored once and reused every epoch, so modifying one in place (e.g. `item["left"].reverse()`) would silently corrupt later epochs.

---

## 2. `LengthBucketBatchSampler` (`sampler.py`): which examples form a batch

**Job:** decide **which example indices go together in one batch**, and in which order the batches come. It works only with indices `0 … len(dataset)-1`; it never builds tensors.

**Why not plain shuffling:** every batch is padded to its longest input, so mixing a 3-token "yeah" with a 60-token sentence wastes most of the batch on padding. Putting examples of **similar length** together keeps padding small.

**Length of an example** = number of **characters** of `target_text` + every string in `left` and `right`. Characters, not tokens: the length is then the same for every model, so with the same seed every model gets the same batches.

**Logic of `__init__(dataset, batch_size, chunk_batches, shuffle, seed=0)`.** All batches are built here, **once**:
1. **Check** that `batch_size` and `chunk_batches` are ≥ 1.
2. **Create the sampler's own random generator,** seeded with `seed`. It's independent of PyTorch's global random state, which the model's initialisation also uses.
3. **Compute every example's length once.**
4. **Train mode** (`shuffle=True`):
   1. **Shuffle all indices.**
   2. **Cut them into chunks** of `chunk_batches × batch_size` indices (e.g. 50 × 32 = 1,600). Only the last chunk can be shorter.
   3. **Sort each chunk by length.**
   4. **Cut each sorted chunk into batches** of `batch_size`.
   5. **Shuffle the order of the batches.**

   **Why chunks:** a full sort over the whole split would put all the short segments together, and those are mostly backchannels ("yeah", "uh-huh"). Batches would become almost single-class. Sorting only inside random chunks keeps batches mixed while padding stays small.
5. **Eval mode** (`shuffle=False`): sort all indices by length, **longest first**, then cut into batches. There's no gradient, so no class-mix concern. Longest first means an out-of-memory error shows up in the first batch, not after minutes of evaluation.
6. **Store the batches.**

**`__iter__`** yields the stored batches, one list of indices at a time. **`__len__`** returns their number. It's exact, which matters because the LR schedule counts its steps from it.

**Consequences:**
- **The same batches, in the same order, every epoch.** This is a deliberate choice: it's simpler, and resuming reproduces the same batches from the seed.
- **At most one batch per epoch is smaller than `batch_size`** (the end of the last chunk).

---

## 3. `SegmentCollator` (`collator.py`): one batch of examples → model-ready tensors

**Job:** the DataLoader hands it **one batch**: the list of items whose indices the Sampler grouped. It returns the tensors the model takes. This is the **model-dependent** piece: it holds the model's tokenizer, and each family builds its own.

**Settings** (`__init__(tokenizer, max_length, separator, markers=("[T]", "[/T]"), suffix="")`):
- **`tokenizer`:** the model's own, loaded from the same checkpoint name as the model.
- **`max_length`:** the token limit per input.
- **`separator`:** the text between two segments (`"\n"` for now; DESIGN.md Part 2 explains why this is still open).
- **`markers`:** the text wrapped around the target. Spaces, if wanted, are part of the markers: `("[T] ", " [/T]")`.
- **`suffix`:** text added after the last segment. **Decoder only**; `""` for the encoder.

It also keeps two counters: `n_items` and `n_truncated`.

**Logic of `render(item)`: one item → one string:**
1. Take `left` **reversed** (back to speaking order).
2. Add the target wrapped in the markers.
3. Add `right`.
4. Join them with the separator, and append the suffix. The suffix is glued on, without a separator.

Example (k = 2, `separator="\n"`, `markers=("[T] ", " [/T]")`, encoder):
```
Some in some introductions are in order .
Oh , OK
[T] sorry . [/T]
OK .
Getting ahead of myself .
```
For the decoder, `"\nLabel of [T]:"` would follow as a last line. An empty `left` or `right` adds nothing, so there's no stray separator.

**Logic of `__call__(items)`: one batch → tensors:**
1. **Render every item.**
2. **Tokenize all the strings in one call:**
   - `padding="longest"`: pad to the longest input in *this* batch;
   - `truncation=True` with `max_length`: cut from the end when an input is too long;
   - `return_token_type_ids=False`: one sequence per input, so they would carry no information;
   - `return_tensors="pt"`.
   The tokenizer adds the model's special tokens itself (`[CLS]`, `[SEP]` for DeBERTa).
3. **Count truncations.** An input can only have been cut if it now fills `max_length` exactly; only those are tokenized again without truncation to check. `n_truncated` and `n_items` are updated.
4. **Labels:** one per item (sequence classification: one class for the whole input), as a `[B]` tensor. Test items (`label=None`) get `-100`, which PyTorch's cross-entropy ignores.
5. **Return:**
   - `input_ids` `[B, L]`;
   - `attention_mask` `[B, L]` (1 = real token, 0 = padding);
   - `labels` `[B]`;
   - `seg_ids`, a Python list. Tensors can't hold strings, and the seg_ids are needed to write predictions.

**No `if encoder / if decoder` branches:** everything that differs between families is a setting (separator, markers, suffix) or the tokenizer passed in.

---

## 4. `build_dataloader` (`loader.py`): wiring the pieces together

**Job:** build the three pieces from one set of arguments and connect them in a standard PyTorch `DataLoader`:
```python
dataset  = SegmentDataset(split_dir, past_k, future_k, allow_test=allow_test, keep_labels=keep_labels)
sampler  = LengthBucketBatchSampler(dataset, batch_size, chunk_batches, shuffle, seed=seed)
collator = SegmentCollator(tokenizer, max_length, separator, markers=markers, suffix=suffix)
return DataLoader(dataset, batch_sampler=sampler, collate_fn=collator, num_workers=0, pin_memory=torch.cuda.is_available())
```

**What the DataLoader does each step:**
1. Ask the sampler for the next batch of indices.
2. Fetch those items from the Dataset.
3. Pass the list to the Collator.
4. Hand the result to the training loop.

**Settings:**
- **`batch_sampler=`** replaces `batch_size` and `shuffle`: the sampler already decides both.
- **`num_workers=0`:** batches are prepared in the main process. That costs ~3 ms per step, negligible next to the model. Worker processes would each hold a *copy* of the Collator, so its truncation counter would stay 0 in the main process.
- **`pin_memory`:** on when a GPU exists, for faster copies to the GPU.
- **The Collator stays reachable** as `dataloader.collate_fn`, e.g. to read `n_truncated` after an epoch.

---

## Checked behaviour

These were verified on the real v4 data:
- **Dataset:** context windows in the right order; empty `left` at meeting starts; 16,188 val items; test refused without the flag and `label=None` with it; `keep_labels` gives 13,087 stage-2 train items with context unchanged.
- **Sampler:** every index exactly once; one short batch per epoch; eval longest first; the same seed gives the same batches.
- **Collator** (with DeBERTa's tokenizer): `[CLS] … [SEP]`, right padding with `[PAD]`, mask 0 on padding, `-100` labels on test, truncation counted.
- **DataLoader:** a full val epoch gives 506 batches and every seg_id exactly once.
