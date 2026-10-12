# A — Data transformation

## Goal

Build a validated, versioned C3 dataset release from the C2 annotated rows and
links. Keep meeting-level splits and folds reproducible, and provide a formatter
that exposes context without leaking gold labels or links into model input.

## Status

**v1 remains frozen; v2 is rebuilt from the updated Task 2 export.** Split
utilities, context formatting, release loading and validation, and the release
builder are implemented. The local ICSI Core NXT archive matches the SHA-256
recorded in [source_archive.json](provenance/source_archive.json). The
reproducible 51/12/12 meeting split and its five folds are recorded in
[split_v1.json](provenance/split_v1.json). This is a project-selected
replacement for the unavailable Table 9 list, not a transcription of that
list.

The original frozen release remains at `data/v1/` with fingerprint
`1b8b1fbc27e70c81e5ff7c710380138f1b05db7f2104e174e3a7141e2345c5bb`. The
updated release is at `data/v2/`, validates with 118,694 rows and 3,831 links,
and has fingerprint
`bcf389dd0bee019de4fdca20c270fd083e3c87e7785ee24781291c435d700d3f`. Its
input/output hashes are in [release_v2.json](provenance/release_v2.json).

The new Task 2 rows export contained 115,874 rows and omitted 2,820 IDs from
C1. The overlapping rows matched frozen v1 exactly on all fields supplied by
Task 2. None of the omitted rows was a Layer-2 proposal or had a link, so those
context rows and omitted annotation fields were recovered from frozen v1 to
preserve complete meeting streams. The new Task 2 links file was used as
provided; it changes the responder chain for two proposals in meeting Bro005
by removing one response segment from each chain. The full completion method
and all input/output hashes are in
[c2_v2_completion.json](provenance/c2_v2_completion.json). The Task 2 generator
is still unavailable, so the source files are fingerprinted but cannot be
regenerated independently.

Both release folders are versioned on this branch. Their large `rows.jsonl`
files use Git LFS; teammates can retrieve them with `git lfs pull`. Raw C1/C2
source inputs remain local and are not included in the branch.

The C1 parser was regenerated from the `A/nxt-parser` branch using the verified
NXT archive and its transcript inputs. The generated 118,694-row file matches
the branch's committed C1 fingerprint. See [C1 provenance](provenance/c1.json)
and the [parse validation report](reports/parse_validation.md).

The plan's expected 321 `no_word_ref` source elements appear as 322 rows because
`Bed006.A.dialogueact1207` is split into two parts; both parts have no word
reference. Both releases record 5 malformed link annotations, which remain
excluded from Layer-2 gold by the schema. The releases use the documented
D1-D16 defaults. D10 is the user-directed replacement split described above;
use a future release if the project lead supplies the official Table 9
assignments.

## Decisions

- **D10 — Split and folds:** 51 train, 12 validation, and 12 test meetings;
  assignment is stratified by meeting-series prefix and ordered by seeded
  SHA-256 (`icsi-mrda-split-v1`). Folds use the existing deterministic
  round-robin method over train and validation meetings. This user-directed
  replacement preserves whole meetings and uses the 51/12/12 size from a
  [published MRDA setup](https://aclanthology.org/2023.findings-emnlp.505/);
  it is not the lead's Table 9 membership list. See the
  [split provenance schema](schema/03_release.md#split-provenance).

## Create or verify the split

From the repository root, regenerate the same assignment with:

```powershell
$env:PYTHONPATH = "src"
python scripts/make_splits.py
```

The input is `data/ICSI-metadata.xml`; the output is
`tasks/A-data-transformation/provenance/split_v1.json`.

## Verify the code

From the repository root, run the fixture checks with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests/data_release -v
```

## Build a release

The build script requires complete C2 rows and links,
`tasks/A-data-transformation/provenance/split_v1.json`,
source archive metadata, and a JSON file containing D1 through D16 defaults.
It validates the inputs, writes five files under `data/vN/`, then records the
release fingerprint and hashes in `provenance/release_vN.json`. Keep frozen
release folders unchanged; later data changes get the next version. Use
`--dry-run` only with fixture inputs and a temporary output directory.

For the current v2 inputs, first complete the partial Task 2 rows export using
the still-frozen v1 rows as a recovery source, then build v2:

```powershell
python scripts/complete_c2_rows.py `
  --task2-rows data/interim/annotated/rows.jsonl `
  --previous-release-rows data/v1/rows.jsonl `
  --c1-rows data/interim/parsed/rows.jsonl `
  --links data/interim/annotated/links.jsonl `
  --output data/interim/normalized/v2_rows.jsonl `
  --provenance tasks/A-data-transformation/provenance/c2_v2_completion.json

$env:PYTHONPATH = "src"
python scripts/make_release.py `
  --rows data/interim/normalized/v2_rows.jsonl `
  --links data/interim/annotated/links.jsonl `
  --splits tasks/A-data-transformation/provenance/split_v1.json `
  --source tasks/A-data-transformation/provenance/source_archive.json `
  --decisions tasks/A-data-transformation/provenance/decision_defaults_v1.json `
  --output data/v2 `
  --version v2 `
  --provenance tasks/A-data-transformation/provenance/release_v2.json `
  --provenance-input tasks/A-data-transformation/provenance/c2_v2_completion.json
```

## Decisions and experiments

Record approved D1–D16 values in the release decisions input. Keep experiment
notes under `experiments/` and link their conclusions here when they inform a
release decision.
