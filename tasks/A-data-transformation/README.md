# A — Data transformation

## T1: NXT ingestion

Status: implemented. The current corpus parses cleanly after a documented reconstruction of the missing pipe-preserving transcript record for `Bro003.F.dialogueact12`.

## Decisions

- A1 split boundaries require the exact meeting, channel, and source timestamps, an exact transcript/source part-count match, and one unique ordered lexical boundary; reason: ambiguous boundaries must not become fabricated gold data. See [the C1 schema](schema/01_parsed_rows.md#pipe-boundary-acceptance-rule).
- A1 uses transcript-aligned word boundaries for part timing and proportional timing only when a boundary word lacks usable timing; reason: preserve source timing where available while keeping split spans inside the source act. See [timing methods](schema/01_parsed_rows.md#timing-methods).
- NXT dialogue-act and word XML remain canonical; transcript text is used only to locate pipe boundaries and is never copied into C1; reason: transcript review must not alter dialogue-act labels or word data.
- Source token text and case, including disfluencies, fillers, fragments, quotes, and non-`w` content, are preserved; reason: C1 should represent the annotated speech rather than a cleaned transcript.
- D13 remains pending: the C1 artifact preserves all NXT lexical and non-lexical elements, and downstream ownership must decide how they are presented to models.
- Transcript-only corrections are recorded in the tracked provenance table and may repair markup, spelling, tokenization, or missing pipe-preserving renderings, but may not rewrite NXT source data; reason: corrections must be reproducible without changing the gold source.
- The missing `Bro003.F.dialogueact12` transcript record is reconstructed from its canonical NXT lexical word range with the pipe after the initial `Uh`; reason: downstream stages need a usable deterministic split, while the provenance record makes clear that this is a reconstruction rather than recovered original transcript text.
- Unresolved source records are retained for inspection, marked `ref_unresolved`, reported once per source dialogue act, and make the parser exit non-zero; reason: unresolved data must block a frozen release rather than be silently accepted.
- The local development environment uses Python 3.11 with dependencies installed from `requirements.txt` into `.venv`; reason: parser and test execution must be reproducible for teammates.
- Table 5 comparison remains pending because the table is not present in this repository; the measured source counts are recorded in the generated validation report.

## Experiments

| Number | Question | Result | Conclusion |
|---|---|---|---|
| T1-E1 | Does the parser preserve the source structure and produce unique output IDs? | 118,694 generated rows matched the independent pipe-part count; the uniqueness and JSONL round-trip test passed. | The generated C1 structure and deterministic IDs satisfy the current checks. |
| T1-E2 | Why did parse validation report 17 unresolved source dialogue acts? | All source IDs and word references resolved; six lacked pipe-preserving transcript records and eleven had transcript wording/tokenization that prevented unique alignment. | These were transcript/alignment issues, not neglected dialogue acts or label decisions. |
| T1-E3 | Can safe transcript-side inconsistencies be corrected without changing NXT data? | Seventeen reviewed corrections are tracked in provenance, including the reconstructed Bro003 record. | Corrections resolve the parse while preserving canonical dialogue-act and word data; the reconstructed record is explicitly documented as synthetic. |
| T1-E4 | Is the repository environment reproducible? | Python 3.11.9 with `pytest==8.3.4` in `.venv`; the parser test file passes 8 tests. | The documented virtual environment is sufficient for the current parser/test scope. |

The original 17 cases were retained, not neglected: their NXT dialogue-act IDs and word ranges resolve. Six had no matching pipe-preserving transcript record, and eleven had a transcript record whose wording or tokenization did not give a unique lexical boundary. Seventeen reviewed transcript-only corrections are now tracked in [transcript_corrections.json](provenance/transcript_corrections.json). The `Bro003.F.dialogueact12` record is explicitly reconstructed from canonical NXT lexical words because its original transcript record is absent; it is not presented as recovered source text.

See the [parser usage guide](parser_usage.md), [the NXT concept note](../../concepts/nxt-icsi-annotations.md), [source provenance](provenance/source_archive.json), and the generated `reports/parse_validation.md` after running `python scripts/parse_nxt.py`.
