# A — Data transformation

Part A of *Structured Meeting Intelligence*. It turns the ICSI MRDA corpus (NXT release) into a
validated, versioned dataset: one row per dialogue act with a class, the act's own text,
proposal-to-reply links, leak-free meeting splits, and a formatter that builds model input.
Short version: [SUMMARY.md](SUMMARY.md).

1. [Overview](#1-overview) · 2. [Why](#2-why) · 3. [What each task did](#3-what-each-task-did) ·
4. [Deviations from the plan](#4-deviations-from-the-original-plan) · 5. [Decisions and choices](#5-decisions-and-choices) ·
6. [How to use the pipeline](#6-how-to-use-the-pipeline) · 7. [Inputs and outputs](#7-inputs-and-outputs) ·
8. [Tests and validation](#8-tests-and-validation) · 9. [Results](#9-results) · 10. [Limitations](#10-limitations-and-open-items) ·
11. [Contributions](#11-contributions)

## 1. Overview

Three tasks, one automated chain. One command rebuilds everything from the raw corpus:

```
ICSI_core_NXT.zip + NXT XML (DialogueActs, Words) + MRDA transcripts (*.trans)
   │ Task 1  parse      -> C1 parsed rows            118,694 rows, one per dialogue act
   │ Task 2  annotate   -> C2 rows + links           3,831 proposal records
   │ Task 2  clean      -> drops z, x and empty rows 115,874 rows
   │ Task 3  split      -> 75 meetings -> train / val / test, 5 folds
   │ Task 3  renumber   -> positions 0..n-1 in every meeting
   │ Task 3  release    -> data/v3/  five files, validated, fingerprinted
   ▼
 Task 3 formatter (context windows) and Layer-2 gold view read the release
```

- **Task 1** parsed the NXT annotation files into act-level rows. The boundary between acts
  inside one utterance is not in the NXT files, so it is recovered from the MRDA transcripts.
- **Task 2** interpreted the labels (7 classes, proposals, flags) and linked each proposal to
  its replies using the adjacency-pair annotation. A cleaning step then removes acts that are
  not usable speech.
- **Task 3** defined splits and folds, a validated five-file release, the Layer-2 gold view, and
  the context formatter. It also owns the one-command pipeline (`scripts/run_pipeline.py`).
- **Reproducible.** The same archive and code give byte-identical files. The source archive and
  each stage's output are fingerprinted; the pipeline compares them with the committed values.
- **Tested.** 531 automated tests across the three tasks and the pipeline (Section 8).

Current build: `data/v3`, 115,874 rows, 3,831 links, release fingerprint
`b4a72c6a0a9a173a7680e95f9757bcf3bf32a53850fca6d9aaaaf9c9d4786c9b`. Generated data is never
committed (`data/` is git-ignored); regenerate it with the pipeline.

## 2. Why

Parts B (evaluation) and C (models) need rows with one trustworthy class each, the words of
that act only, the proposal-to-reply structure for linking, splits that never share a meeting
across train and test, and one shared way of building a context window. Part A produces these
once, so every model is trained and scored on identical data.

## 3. What each task did

### 3.1 Task 1 — NXT ingestion (`A/nxt-parser`)

Reads `DialogueActs/*.dialogue-acts.xml`, resolves each act's word range in `Words/*.words.xml`
(`src/smi/data/nxt/`, `scripts/parse_nxt.py`). Output C1: 23 fields per row, defined in
[schema/01](schema/01_parsed_rows.md).

- **One row per dialogue act, not per utterance.** An utterance labelled `fg|s` is two acts
  (7,193 utterances carry a `|`). The NXT files give one word range for the whole utterance, so
  the boundary is located by aligning the MRDA transcript, which places the `|` in the text.
  A boundary is never guessed: if it cannot be placed uniquely, the rows are kept, flagged
  `ref_unresolved`, and the run fails.
- **17 reviewed corrections.** For 17 utterances the transcript record is missing or worded
  differently. Each was fixed by hand in `provenance/transcript_corrections.json`; the table
  only helps locate the boundary and never changes a label, word or time.
- **Source gaps are flagged, not repaired.** Informational flags: `no_word_ref`,
  `type_fallback_original`, `word_time_missing`. Error flags: `ref_unresolved`, `ref_malformed`,
  `time_invalid`, `type_unavailable`. No row has an error flag.
- **Two kinds of time.** `src_start/src_end` are the source's; `start/end` are derived per act
  with a recorded `timing_method` (`element`, `transcript_word_boundary`,
  `transcript_proportional`, `source_proportional`).
- **Identity and order come from the source.** `seg_id = {meeting}-{channel}_{start_ms}_{end_ms}`
  (`_p1`, `_p2` for split acts); `position` is the rank in a fixed order, so neither changes when
  the timing rule does.
- **Text is kept as spoken** (fillers, fragments, case, quotes). Validation report:
  [parse_validation.md](reports/parse_validation.md). Usage guide: [parser_usage.md](parser_usage.md).

### 3.2 Task 2 — Annotation semantics (`A/labels-links`)

`src/smi/labels.py` (the single class list), `src/smi/data/annotate/`, `scripts/annotate_rows.py`,
`scripts/clean_rows.py`. It adds these fields to every C1 row:

| Field | Meaning |
|---|---|
| `label` | One of `cs, co, aa, bk, ar, cc, other` (`aap`→`aa`, `arp`→`ar`) |
| `label_bk_merged` | Same, with the bare backchannel `b` counted as `bk` |
| `target_tags`, `is_multi_target` | Every target class the act carries, in `LABELS` order; true if two or more |
| `is_layer2_proposal` | `cs` or `co` is among `target_tags` (not decided by `label`) |
| `general_tags`, `disruption_markers` | Analysis-only; `%-` and `%--` are the markers |

- **Class rule.** Only the speaker's own side of a colon counts (manual p. 21). With several
  target classes the row keeps all of them and takes one `label` by `cc > co > cs > ar > aa > bk`.
- **Flags, then cleaning.** Annotation marks `is_nonspeech` (whole label `x`), `is_nonlabeled`
  (`z`) and `is_empty`. The cleaning step then **removes** those rows (2,820) and the flags with
  them, so the released data has none of the three fields.
- **Links.** One record per proposal (`proposal_id`, `responder_chains`, `has_response`,
  `status`, `adjacency_tokens`, `pair_group`). Tokens belong to the whole act; a reply joins a
  chain for its speaker number; all open pairs close when the numbering returns to 1 (506 times).
  Status is `ok`, `no_pair`, `ambiguous` or `malformed`; the last two are never turned into
  "no reply".
- **Safety.** `check_contract` runs before anything is written (parsed fields unchanged, one link
  per proposal, chains point at real rows, error rows never flagged). Reports:
  [annotation_report.md](reports/annotation_report.md).

### 3.3 Task 3 — Splits, formatter and release (`A/splits-formatter-release`)

`src/smi/data/splits.py`, `release.py`, `src/smi/inputs/formatter.py`, `scripts/make_splits.py`,
`scripts/make_release.py`, and later `scripts/run_pipeline.py`. Documented in
[schema/03](schema/03_release.md).

- **Splits and folds.** Whole meetings only: 51 train, 12 val, 12 test, stratified by meeting
  series, ordered within a series by seeded SHA-256 (seed `icsi-mrda-split-v1`). Folds: train and
  val meetings sorted, assigned round-robin to 5 folds; test meetings have `fold = null`. Every
  meeting's split, fold and reason is in [split_v1.json](provenance/split_v1.json).
- **Release.** Exactly five files: `rows.jsonl`, `links.jsonl`, `splits.json`, `metadata.json`,
  `checksums.sha256`. The checksum file hashes the other four and never itself; its own SHA-256 is
  the **release fingerprint**. Nothing in the files depends on time, machine or path.
- **Loader and validator.** `load_release()` requires the exact file set, verifies every
  checksum, then runs over 60 checks (unique `seg_id`, labels in `LABELS`, one split per
  meeting, folds in range, links pointing at real rows, no error flags, contiguous positions).
- **Layer-2 view.** `acceptable_responses(record, responders="all", segments="first")` and
  `ReleaseData.layer2_gold()`. Gold holds only `ok` and `no_pair` records; `malformed` and
  `ambiguous` are kept in the file but never become a "none" target.
- **Formatter.** `ContextFormatter.select_context(seg_id, k_left, k_right)` returns row ids;
  `format(seg_id, k_left, k_right, speaker_mode)` renders them with the target in `[T]…[/T]`.
  Speaker modes: `raw`, `relative` (SAME/OTHER), `window_letters` (A, B, …). The window never
  leaves the meeting; output never contains a label, tag, link, split or fold.
- **History.** v1 (6 Oct) and v2 (7 Oct, built from the cleaned Task 2 export with the removed rows
  restored from v1) were built and later removed. The adaptation to cleaned data and the pipeline came after (Section 4).

## 4. Deviations from the original plan

Against `PART_A_TASK_BREAKDOWN_v2`. Reasons in Section 5 where a decision was involved.

| Plan said | What happened | Why |
|---|---|---|
| Input is the NXT archive only | A fourth input is required: MRDA transcripts (`transcripts/*.trans`) | The `\|` boundary between acts is not in the NXT files |
| 321 rows flagged `no_word_ref` | 322 rows (321 acts; one act has two parts) | A split act with no words gives a row per part |
| `:` meaning unverified; example `s^cs:s^co`→`co`; 22 two-target acts | Left of the colon only, so `s^cs:s^co`→`cs`; 18 two-target acts | Labelling manual p. 21; matches the specification's 18 |
| `z` and `x` both set `is_nonspeech` | `x` (whole label) sets `is_nonspeech`; `z` sets a new `is_nonlabeled`; an appended `x` (`s.x`) stays speech | Manual pp. 17, 19–20, 113–114; most `z` rows contain words |
| Every row stays in the file; formatter drops or replaces flagged rows | A separate cleaning step removes z, x and empty rows (2,820); the three flags are gone; the formatter has no `nonspeech` option or `[nonverbal]` placeholder | Meeting decision: drop them once, upstream |
| `position` contiguous over all rows | Cleaning keeps original positions (gaps); the pipeline then renumbers 0..n-1 per meeting before the release; `seg_id` is never renumbered | Keeps cleaning's fingerprints stable and the validator's contiguity rule intact |
| Splits from the lead's Table 9 lists | Project-selected 51/12/12 split (`lead_table_9: false`) | Table 9 was never supplied. A new release version is needed if it arrives |
| Freeze `data/v1`; `release_v1.json` with commit, date and Class P confirmations | v1 and v2 built, then removed. Current `data/v3` comes from the pipeline; its record is `build_manifest.json` (no commit or date by design). No `release_v3.json`; Class P confirmations not recorded | No freeze gate was run; Table 9 and lead sign-off are open |
| Three scripts: parse, annotate, make_release | Added `scripts/run_pipeline.py` (one command, fail-fast checks, fingerprint comparison) | Fewer manual steps, same result every time |
| `complete_c2_rows.py` rebuilds v2 from v1 | Obsolete; kept in the repo but cannot run without v1 | The new chain releases the cleaned rows directly |
| New `decision_defaults_v3.json` | D5, D12 and D15 reworded for the cleaned design; stored in each release's `metadata.json` | The v1 wording ("keep every row") would be false |
| Not delivered | `schema/02`, `concepts/mrda-tags-and-adjacency-pairs.md`, `pyproject.toml`, root README row, Table 5 comparison, the "every field documented once" check | Open (Section 10) |

## 5. Decisions and choices

Sources in order of authority: the labelling manual (Dhillon et al., 2004), the corpus paper
(Shriberg et al., 2004), the earlier exploratory report, then measurement. Where none decides, the
choice is called a convention. † marks figures from the October progress report's exploratory
checks, which are not yet reproduced by a script.

**Task 1**

| Decision | Reason |
|---|---|
| Annotation files are the only source of labels and words; transcripts only locate boundaries | The transcripts differ in wording; this keeps every row traceable to the annotation |
| The unit is the dialogue act (split on `\|`) | Each row then has one label and exactly its own words |
| Only `\|` is interpreted in parsing; `^`, `.`, `:` pass through unchanged | What they mean for the classes is a labelling decision (Task 2) |
| Never guess a boundary; record unresolved cases as explicit corrections | A wrong boundary looks valid but carries the wrong words |
| Text kept as spoken | Normalisation is a modelling choice and cannot be undone |
| Source and derived times stored separately, with the method | Changing the derivation never alters the original data or a row's identity |
| Anomalies flagged (informational or error), never turned into normal rows | Later steps exclude knowingly, not through invisible repairs |

**Task 2**

| Decision | Reason |
|---|---|
| Only the speaker's own side of a colon counts | Manual p. 21; gives exactly the specification's 18 two-target acts (22 if pooled†) |
| Seven classes; partial accept/reject merged into accept/reject | Required by the specification; manual pp. 6–7, 59, 62 |
| All target classes kept; one `label` by `cc > co > cs > ar > aa > bk` | Tag order is alphabetical (p. 15), so it cannot decide; the order is a convention that affects 18 rows and is reversible because the full set is kept |
| `%-` and `%--` are disruption markers; `%` is an ordinary tag | Manual pp. 110–111 |
| Non-speech only if the whole label is `x`; `z` has its own flag | `x` with a suffix keeps its act (pp. 19–20); `z` is "unlabelled", not silent: 2,422 of 2,442 rows contain words† |
| Error rows are never empty, non-speech or unlabelled | A failed reference must not pass for data |
| Proposal = `cs` or `co` among the target classes | Independent of the class order |
| Adjacency tokens belong to the whole act | Manual p. 27; token count equals part count in only 29.4% of split acts† |
| Close all open pairs when numbering returns to 1 | Manual p. 26 says numbers never repeat, but the data restarts 506 times; this cut 326 links over 100 rows away to 9† |
| One status per proposal; a broken token makes it `malformed` | A parse failure must not be reported as "no reply" |
| Rows read in `position` order, never by derived times | A change to the timing rule cannot change who is a neighbour |
| Cleaning is a separate, checked step; rows are not renumbered by it | Meeting decision; `check_cleaned` proves cleaned equals annotated minus dropped |

**Task 3**

| Decision | Reason |
|---|---|
| Whole meetings per split; stratified by series; seeded SHA-256 order | No meeting in two splits; every series represented; reproducible |
| 5 folds, round-robin over sorted train+val meetings; test fold is null | Explicit and stable; the test set is never in a fold |
| Release rows come from the cleaned output, positions renumbered | Releases carry no flags and no gaps |
| Checksums over four files; fingerprint = hash of the checksum file | A file cannot list its own hash; one value identifies the release |
| No timestamps, commits or paths in release files | The same input must give the same bytes on any machine |
| Gold = `ok` + `no_pair`; acceptable set = first segment of every chain | One answer per proposal; malformed/ambiguous are never silently "none" |
| Context unit is one row; stream is the meeting in `position` order | "Next 5 acts" means the same everywhere |
| Speaker modes are capabilities; default `window_letters`; no benchmark mode mandated | Part A requires no cross-mode comparison |
| Validator rejects instead of repairing | A silently wrong dataset is worse than a failed build |

## 6. How to use the pipeline

**Requirements.** Python 3.11 (the project standard) or newer, tested on 3.12; the pipeline uses the
standard library only, and `pip install -r requirements.txt` adds pytest for the tests. A full build
takes about 2 minutes and 650 MB of disk (510 MB interim, 130 MB release).

**Inputs.** Three things, all local (never committed):

```
ICSI_core_NXT.zip                  sha256 cf4860245b9ca9c9ed11a66e5a74cfea2a99a686a52f7580aa0a6bc2225895a9
data/raw/icsi_core_nxt/            (or any folder; pass --raw-root)
+-- DialogueActs/                  465 *.dialogue-acts.xml   (from ICSI/DialogueActs in the zip)
+-- Words/                         494 *.words.xml           (from ICSI/Words in the zip)
+-- transcripts/                   75 *.trans, one per meeting (MRDA transcripts)
```

The archive is checked against `provenance/source_archive.json`; a different archive stops the
run. The archive also supplies the meeting list (`ICSI/ICSI-metadata.xml`) for the splits.

**Run.**

```bash
python scripts/run_pipeline.py \
    --raw-root data/raw/icsi_core_nxt \
    --archive  data/raw/ICSI_core_NXT.zip \
    --version  v3
```

| Option | Default | Meaning |
|---|---|---|
| `--raw-root` | `data/raw/icsi_core_nxt` | Folder with `DialogueActs`, `Words`, `transcripts` |
| `--archive` | `data/raw/ICSI_core_NXT.zip` | The hash-checked archive |
| `--data-root` | `data` | Receives `interim/` and the release folder |
| `--version` | `v3` | Release folder name (`v1`, `v2`, …); must not already contain files |

It prints one line per stage, then compares each rebuilt file with the committed fingerprints in
`provenance/`. Exit code 0 means success; 1 means it stopped, with the reason on stderr.
Checks run **before** any long stage: version label, input folders, archive, provenance files, and
that the release folder is free. Tracked provenance and reports are read, never written.

| Stops with | Meaning and fix |
|---|---|
| `has no DialogueActs/ folder` | Wrong `--raw-root`, or the layout above is incomplete |
| `hashes to … but provenance/source_archive.json records …` | Not the project's archive; download it again |
| `already exists; delete it or choose another --version` | The release folder has files; the pipeline never overwrites a release |
| `parsing found fatal source or contract errors; see …` | Open `interim/reports/parse_validation.md` |
| `annotation broke the C2 contract` / `cleaning broke its contract` | A stage's own check failed; the message lists the problems |

**What it writes.**

```
data/interim/   parsed/rows.jsonl   annotated/{rows,links}.jsonl   cleaned/{rows,links}.jsonl
                release_input/rows.jsonl   splits/{ICSI-metadata.xml,split.json}
                reports/{parse_validation,annotation_report,cleaning_report}.md
                build_manifest.json        every file's hash, line count, removed-row count
data/v3/        rows.jsonl  links.jsonl  splits.json  metadata.json  checksums.sha256
```

**Check it worked.** On the current code and archive the pipeline reports `matches` for the parsed
rows (`c1.json`) and the cleaned rows and links (`c2_clean.json`), "same assignment as committed
split_v1.json", and the release fingerprint above. The annotated-file comparison is skipped while
`c2.json` is unreadable (Section 10); its hashes are `cce142e5…` (rows) and `bf6d20b7…` (links).
A mismatch is a defect to investigate, not a warning to ignore.

**Read the data.**

```python
from smi.data.release import load_release          # run with PYTHONPATH=src
from smi.inputs.formatter import ContextFormatter

release = load_release("data/v3")                  # verifies checksums, validates everything
train = [r for r in release.rows if r["split"] == "train"]       # r["fold"] is 0..4
gold = release.layer2_gold()                       # [{proposal_id, acceptable_responses, proposal, link}]
formatter = ContextFormatter(release.rows)
formatter.format(gold[0]["proposal_id"], 3, 3, "window_letters") # 3 rows each side, target in [T]…[/T]
```

**Rebuild or change a version.** Delete the release folder (it is generated) or pick a new
`--version`. Never edit a release by hand; any change to inputs or rules is a new version.

**Single stages** (`parse_nxt.py`, `annotate_rows.py`, `clean_rows.py`, `make_splits.py`,
`make_release.py`) still exist, but they write into tracked provenance and report files, and
`annotate_rows.py` and `clean_rows.py` record `git rev-parse HEAD`, so they only run inside a git
checkout. Prefer the pipeline. See [parser_usage.md](parser_usage.md) for the parser alone.

## 7. Inputs and outputs

| Item | Location | Notes |
|---|---|---|
| Schemas | `schema/01_parsed_rows.md`, `schema/03_release.md` | `schema/02` is not written yet |
| Provenance | `provenance/` | `source_archive.json`, `c1.json`, `c2_clean.json`, `split_v1.json`, `transcript_corrections.json`, `decision_defaults_v3.json` |
| Reports | `reports/` | `parse_validation.md`, `annotation_report.md` |
| Concept note | `../../concepts/nxt-icsi-annotations.md` | NXT structure |
| Generated data | `data/` | Git-ignored |

## 8. Tests and validation

531 tests in `tests/`, run with `python -m pytest tests -q`. Fixtures are small hand-written
files, never corpus excerpts.

| Folder | Tests | Covers |
|---|---|---|
| `tests/data_nxt` | 64 | Parsing a hand-built NXT corpus, every error flag, ids, order, report, serialisation |
| `tests/data_annotate` | 194 | Classes, flags, tags, every link scenario, contract, cleaning, report, scripts |
| `tests/data_release` | 248 | Formatter, splits, over 60 validator checks, release build, Layer-2 view, committed releases |
| `tests/pipeline` | 25 | The whole chain on a 75-meeting toy corpus: counts, identical reruns, refusals |

- Tests that need generated data (`data/vN`, `data/interim`, the raw corpus) skip when it is
  absent, so run the pipeline first to exercise them.
- 4 tests are strict expected failures that mark known problems (Section 10).
- Latest run on the real `data/v3`: 522 passed, 5 skipped, 4 expected failures. Run in a clean
  `git clone` with the pending changes laid over it, only the data-dependent tests skipped. A one-off check with 20
  deliberately injected faults across the three tasks (run before the latest Task 3 changes)
  caught all 20.
- The pipeline was also run from a clean export of the last commit with pending changes laid over
  it; its release was byte-identical to `data/v3`.

## 9. Results

All from this run's generated files (`build_manifest.json`, `reports/`, `metadata.json`).

| | Count |
|---|---|
| Dialogue-act elements → rows | 111,500 → 118,694 (matches an independent count of `\|` parts) |
| Rows by parts | 104,307 one-part, 14,384 two-part, 3 three-part |
| Timing methods | 104,307 element · 14,379 word boundary · 6 proportional · 2 source-proportional |
| Flags | 322 `no_word_ref` · 1 `type_fallback_original` · 56,659 `word_time_missing` · 0 error |
| Two-target acts | 18 (the specification's figure) |
| Removed by cleaning | 2,820 (90 non-speech, 2,442 non-labeled, 324 empty; some rows have two reasons) |
| Released rows | 115,874 — train 79,039 · val 16,289 · test 20,546 |
| Classes (released) | other 96,883 · bk 7,272 · aa 6,332 · cs 3,041 · ar 1,120 · co 790 · cc 436 |
| Links | 3,831: ok 1,589 · no_pair 2,237 · malformed 5 · ambiguous 0 |
| Layer-2 gold | 3,826 records (ok + no_pair) |
| Replying chains per ok record | 1: 1,357 · 2: 195 · 3: 31 · 4: 6 |
| Reply within the next 5 rows | 1,521 of 1,589 (95.7%) |
| Adjacency numbering restarts | 506 |

Cleaning changes classes only slightly (`aa` −7, `bk` −10, `other` −2,803) and changes two link
records, which each lose one reply row. No proposal is removed.

## 10. Limitations and open items

- **Inputs not supplied:** the lead's Table 9 lists and the published Table 5 counts, so the split
  is project-selected and the Table 5 comparison is pending.
- **Not confirmed:** the Class P decisions (D1, D3, D5, D10, D13, D14, the colon reading) are not
  recorded as signed off; no `release_v3.json` exists. D5, D12 and D15 in `decision_defaults_v3.json`
  are new wording awaiting the lead's agreement. Whether the 236 kept rows with an appended `x`
  (`s.x`) are intended should be confirmed.
- **Adjacency weaknesses:** a reply ordered before its opening act is dropped (431 replies find no
  open pair; 154 of them precede their opener†); only restarts to number 1 are detected, and they
  are not periodic†; there is no distance or speaker guard; 8 resolved proposals are answered only
  by the proposer's own speaker id†. 1,589 linked proposals is a lower bound.
- **Missing deliverables:** `schema/02_annotations_links.md` (until it exists, Section 3.2's field
  table is the reference), the MRDA concept note, `pyproject.toml`, the root README row, and the
  check that every field is documented in exactly one schema.
- **Known defects** (the first three are strict expected failures in the tests):
  `build_cleaning_report` miscounts when an early proposal is dropped (not triggered on real data);
  `write_split_provenance` omits `newline="\n"`, so Windows writes CRLF (the pipeline normalises it);
  `provenance/c2.json` is invalid JSON from a merge; `c2_clean.json` names a commit that predates
  `clean_rows.py`; one PDF is tracked by accident; `complete_c2_rows.py` is obsolete.
- **Environment:** the two raw-corpus tests look for the data at `data/raw/icsi_core_nxt`; data
  tests skip on a machine without generated data.

## 11. Contributions

One subsection per contributor, written by that person (team rule).

### Mohammed Adel — Task 1 (NXT ingestion)
*To be written.*

### Ahmed Yasser — Task 2 (annotation semantics, cleaning)
*To be written.*

### Moamen Talaat — Task 3 (splits, formatter, release)
*To be written.*

## Key files

- [SUMMARY.md](SUMMARY.md) · [parser_usage.md](parser_usage.md) · [schema/01](schema/01_parsed_rows.md) · [schema/03](schema/03_release.md)
- [scripts/run_pipeline.py](../../scripts/run_pipeline.py) · [src/smi/data/release.py](../../src/smi/data/release.py) · [src/smi/inputs/formatter.py](../../src/smi/inputs/formatter.py)
- [split_v1.json](provenance/split_v1.json) · [decision_defaults_v3.json](provenance/decision_defaults_v3.json) · [parse_validation.md](reports/parse_validation.md) · [annotation_report.md](reports/annotation_report.md)
