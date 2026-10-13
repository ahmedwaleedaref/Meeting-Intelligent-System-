# A — Data transformation: summary

Full version: [README.md](README.md). If this file and the README disagree, the README is right.

## TL;DR

Part A turns the ICSI MRDA corpus (NXT release) into a validated, versioned dataset: one row per
dialogue act with a class, its own text, proposal-to-reply links, leak-free meeting splits, and a
formatter that builds model input ([overview](README.md#1-overview)).

- **Task 1** parses the NXT files into 118,694 act-level rows (111,500 source acts). The boundary
  between acts inside one utterance is recovered from the MRDA transcripts.
- **Task 2** maps labels to 7 classes, flags proposals, links 3,831 proposals to their replies, and
  removes 2,820 non-usable rows (`z`, `x`, empty) in a separate cleaning step.
- **Task 3** splits 75 meetings (51 / 12 / 12, five folds), builds a five-file release with
  checksums and a fingerprint, and provides the Layer-2 gold view and the context formatter.
- One command rebuilds everything, byte-identically, in about 2 minutes.

## Run

```bash
python scripts/run_pipeline.py --raw-root data/raw/icsi_core_nxt \
    --archive data/raw/ICSI_core_NXT.zip --version v3
```

Needs the NXT archive, `DialogueActs/`, `Words/` and the MRDA `transcripts/` folder
([inputs and options](README.md#6-how-to-use-the-pipeline)). Output: `data/interim/` and `data/v3/`.
Read it with `load_release("data/v3")` and `ContextFormatter(release.rows)`.

## Key results

| | |
|---|---|
| Released rows / links | 115,874 / 3,831 (ok 1,589 · no_pair 2,237 · malformed 5) |
| Rows by split | train 79,039 · val 16,289 · test 20,546 |
| Two-target acts | 18, as the specification states |
| Release fingerprint | `b4a72c6a0a9a173a7680e95f9757bcf3bf32a53850fca6d9aaaaf9c9d4786c9b` |
| Tests | 543; latest run 538 passed, 5 skipped, none expected to fail ([tests](README.md#8-tests-and-validation)) |

## Key decisions

All in [Section 5](README.md#5-decisions-and-choices); departures from the plan in
[Section 4](README.md#4-deviations-from-the-original-plan).

- Only the speaker's own side of a colon counts (manual p. 21).
- `x` (whole label) is non-speech, `z` is "unlabelled"; both, and empty rows, are removed once,
  upstream. The released data has none of the three flags and the formatter has no non-speech option.
- Several target classes: all kept, one `label` by `cc > co > cs > ar > aa > bk`.
- Adjacency pairs close when the numbering returns to 1; tokens belong to the whole act.
- Splits are project-selected (no Table 9): whole meetings, stratified by series, seeded.
- Cleaning leaves position gaps; the pipeline renumbers positions 0..n-1 before the release.

## Limitations

Details in [Section 10](README.md#10-limitations-and-open-items).

- Table 9 and Table 5 were never supplied; Class P decisions and the new D5 wording are unconfirmed.
- Replies ordered before their opening act are lost, so the link count is a lower bound.
- Missing: `pyproject.toml`, the root README row, `release_v3.json`.
- Known issue: the standalone Task 2 scripts need a git checkout (the pipeline does not).
