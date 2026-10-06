# A — Data transformation

## Goal

Build a validated, versioned C3 dataset release from the C2 annotated rows and
links. Keep meeting-level splits and folds reproducible, and provide a formatter
that exposes context without leaking gold labels or links into model input.

## Status

**Release built; C2 lineage limitation recorded.** Split utilities, context
formatting, release loading and validation, and the release builder are
implemented. The local ICSI Core NXT archive matches the SHA-256 recorded in
[source_archive.json](provenance/source_archive.json). A reproducible 51/12/12
meeting split and its five folds are recorded in [split_v1.json](provenance/split_v1.json).
This is a project-selected replacement for the unavailable Table 9 list, not
a transcription of that list.

The release at `data/v1/` validates with 118,694 rows and 3,831 links. Its
fingerprint is `1b8b1fbc27e70c81e5ff7c710380138f1b05db7f2104e174e3a7141e2345c5bb`;
input and output hashes are in [release_v1.json](provenance/release_v1.json).
The supplied C2 files pass release validation and their hashes/counts are in
[c2.json](provenance/c2.json). Task 2 clean regeneration remains unverified:
the available `A/labels-links` branch contains no annotation generator, so the
C2 provenance records the input fingerprints without claiming how they were
produced.

This branch is configured to include the release outputs; the large
`rows.jsonl` file uses Git LFS. After this branch is published, teammates can
retrieve it with `git lfs pull`. The original C1/C2 and raw source data remain
local and are not included in the branch.

The C1 parser was regenerated from the `A/nxt-parser` branch using the verified
NXT archive and its transcript inputs. The generated 118,694-row file matches
the branch's committed C1 fingerprint. See [C1 provenance](provenance/c1.json)
and the [parse validation report](reports/parse_validation.md).

The plan's expected 321 `no_word_ref` source elements appear as 322 rows because
`Bed006.A.dialogueact1207` is split into two parts; both parts have no word
reference. The release records 5 malformed link annotations, which remain
excluded from Layer-2 gold by the schema. The release uses the documented
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

## Build v1 when inputs are ready

The build script requires the C2 rows and links,
`tasks/A-data-transformation/provenance/split_v1.json`,
source archive metadata, and a JSON file containing D1 through D16 defaults.
It validates the inputs, writes the five release files under `data/v1/`, then
records the release fingerprint and hashes in
`tasks/A-data-transformation/provenance/release_v1.json`.
Use `--dry-run` only with fixture inputs and a temporary output directory.

## Decisions and experiments

Record approved D1–D16 values in the release decisions input. Keep experiment
notes under `experiments/` and link their conclusions here when they inform a
release decision.
