# Schema 03 — Release and formatter output

## Context formatter output

`ContextFormatter` is constructed from a fixed snapshot of C3 rows. Its
`select_context(seg_id, k_left, k_right, nonspeech="drop")` method returns
selected `seg_id` values in meeting `position` order. Its
`format(seg_id, k_left, k_right, speaker_mode, *, nonspeech="drop")` method
renders those rows for a model.

Each selected row is rendered on one line. Lines are joined by a single LF
(`\n`), with no trailing LF. The target text is enclosed by `[T]` and `[/T]`.
When both context sizes are zero, the target is rendered without a speaker
prefix in all speaker modes.

When at least one context size is nonzero, each line uses this layout:

```text
Speaker {speaker_tag}: {content}
```

The speaker tag is the raw `speaker_id` in `raw` mode, `SAME` or `OTHER`
relative to the target speaker in `relative` mode, or a window-local letter
(`A`, `B`, …) in `window_letters` mode. Window letters are assigned on first
appearance from left to right.

With `nonspeech="placeholder"`, a row whose `is_nonspeech` or `is_empty` flag
is true has content `[nonverbal]`. The target remains selected under both
non-speech policies, including when it is flagged. With `nonspeech="drop"`,
flagged non-target rows are removed before context counts are applied.

Formatted text contains only rendered source text, target markers, optional
speaker prefixes, and the `[nonverbal]` placeholder. Gold labels, tags, links,
split, and fold fields are never emitted.

## Split provenance

`tasks/A-data-transformation/provenance/split_v1.json` records every meeting,
its split, fold, and assignment reason. The current project-selected policy uses
51 train, 12 validation, and 12 test meetings. It balances counts by meeting
series prefix, orders meetings within each series by SHA-256 with the recorded
seed, and keeps each complete meeting in one split. The five folds are assigned
by sorted meeting ID, round-robin over train and validation; test folds are
null. `split_assignment.lead_table_9` is false because these assignments are a
replacement policy, not the unavailable Table 9 list.

## C3 release files

Each `data/vN/` release directory (for example `data/v1/` or `data/v2/`)
contains exactly these five files:

- `rows.jsonl`: every C2 row unchanged, with `split` and `fold` appended.
  Train and validation folds are integers in `[0, k)`; test folds are `null`.
- `links.jsonl`: C2 link records, unchanged.
- `splits.json`: `k`, `grouping: "meeting"`, a `meetings` object mapping each
  meeting ID to `{ "split": ..., "fold": ... }`, and the SHA-256 of the
  committed `tasks/A-data-transformation/provenance/split_v1.json` as
  `source_sha256`.
- `metadata.json`: release version and dataset provenance/statistics. Its
  `data_version` matches the release directory (`v1`, `v2`, ...); it also has
  `schema_version`, the source archive name/release/SHA-256,
  row counts by split and label, link counts by status, quality-flag counts,
  the chosen D1–D16 defaults, and CC BY 4.0 attribution. It has no build
  timestamp, commit, machine path, or host.
- `checksums.sha256`: SHA-256 entries for the other four files only, sorted by
  filename. Each line is `<64 lowercase hex characters><two spaces><filename>`.

After a real build, `tasks/A-data-transformation/provenance/release_vN.json`
records the release fingerprint, hashes for the five release files, and hashes
for the C2 rows, C2 links, split, source archive metadata, decision-default,
and any additional provenance inputs. Fixture dry runs do not write this record.

The release loader verifies this exact file set and all four checksums before
validating the row, link, and split contracts. It accepts version labels such
as `v1` and `v2`. It rejects duplicate segment
IDs, unknown labels, inconsistent meeting assignments, missing train/validation
folds, unresolved link IDs, and error-class quality flags. Each
`is_layer2_proposal` row must have exactly one link record, and no ineligible
row may have one.

## Layer-2 gold view

The stored `responder_chains` remain unchanged. The default benchmark acceptable
set uses every responder chain and the first segment of each chain. The
`responders="first"` option selects the first chain (chains are stored in order
of first response position); `segments="any"` includes each segment in the
selected chain or chains. Only `ok` and `no_pair` records enter Layer-2 gold.
`malformed` and `ambiguous` records are excluded, not treated as a `none` gold
target. A `no_pair` record has an empty acceptable set.
