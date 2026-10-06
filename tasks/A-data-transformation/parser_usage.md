# NXT parser usage guide

This guide explains how to run the T1 NXT parser and inspect its output.
Run the commands from the repository root:

```text
C:\Users\mohammed adel\my-github\Meeting-Intelligent-System-
```

## 1. Prepare the environment

The project uses Python 3.11 and a repository-local virtual environment.
Create it once, then install the pinned test dependency:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The parser itself uses Python's standard library. `pytest` is installed so
the parser tests can be run from the same environment.

## 2. Check the input layout

By default, the parser expects the local NXT release at:

```text
data\raw\icsi_core_nxt\
├── DialogueActs\
├── Words\
└── transcripts\
```

The directories must contain the corresponding NXT dialogue-act XML files,
word XML files, and transcript files. The parser raises an error if any of
these three directories is missing.

## 3. Run the parser

Use the default input and output locations:

```powershell
.\.venv\Scripts\python.exe scripts\parse_nxt.py
```

The default output is:

```text
data\interim\parsed\rows.jsonl
```

The parser also writes or updates:

```text
tasks\A-data-transformation\reports\parse_validation.md
tasks\A-data-transformation\provenance\c1.json
```

The tracked transcript-only corrections are loaded automatically from:

```text
tasks\A-data-transformation\provenance\transcript_corrections.json
```

These corrections are used only for transcript pipe-boundary alignment. They
do not modify the NXT dialogue-act or word XML data.

## 4. Use another raw-data directory

Pass `--raw-root` when the NXT release is stored somewhere else:

```powershell
.\.venv\Scripts\python.exe scripts\parse_nxt.py `
    --raw-root C:\data\icsi_core_nxt
```

The alternate directory must still contain `DialogueActs`, `Words`, and
`transcripts` subdirectories.

## 5. Write to another JSONL file

Pass `--output` to write the generated rows to a different location:

```powershell
.\.venv\Scripts\python.exe scripts\parse_nxt.py `
    --output C:\temp\rows.jsonl
```

The validation report and provenance file remain at their repository paths.

## 6. Understand the exit status

The parser writes inspection output before it evaluates the final status.

- Exit code `0`: the rows and validation checks are safe for the next task.
- Exit code `1`: output was written for inspection, but a fatal source or
  contract problem remains.

An exit code of `1` is intentional when unresolved transcript boundaries,
invalid references, invalid timing, unavailable types, or contract failures
remain. Do not treat that output as a frozen release.

The previously missing `Bro003.F.dialogueact12` record is now represented by
a documented reconstructed rendering derived from its canonical NXT lexical
word range. It is recorded in the provenance correction table and does not
claim to recover the original transcript text. The current corpus therefore
exits with code `0`.

## 7. Read the validation report

Open:

```text
tasks\A-data-transformation\reports\parse_validation.md
```

Check the following sections:

- source structure and row-count reconciliation;
- rows by number of pipe parts;
- quality flags;
- contract checks;
- unresolved source summary;
- unresolved transcript boundaries.

The parser must not guess a split when a transcript pipe cannot be mapped to
one unique ordered lexical boundary. Such rows are retained for investigation
with the `ref_unresolved` quality flag.

## 8. Run the parser tests

Run the focused parser tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\data_nxt\test_nxt_parser.py
```

Run all available tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The full-corpus uniqueness test is skipped automatically when the local NXT
release is not present. When the release is available, it checks the
independent pipe-part count, unique `seg_id` values, and JSONL round-trip
consistency.

## 9. Output guarantees

For each source dialogue-act part, the parser emits one JSON object with the
fields defined in [the C1 schema](schema/01_parsed_rows.md). The parser:

- preserves NXT source text, case, fillers, fragments, quotes, and non-`w`
  content;
- uses source timestamps for IDs;
- keeps rows in deterministic meeting-local order;
- validates field order, timing constraints, positions, and ID uniqueness;
- writes compact, deterministic UTF-8 JSONL with one row per line.

Transcript text is used to locate split boundaries only. It is not copied into
the C1 `text` field.

## 10. Related files

- [C1 schema](schema/01_parsed_rows.md)
- [validation report](reports/parse_validation.md)
- [transcript corrections](provenance/transcript_corrections.json)
- [NXT annotation notes](../../concepts/nxt-icsi-annotations.md)
- [T1 task README](README.md)
