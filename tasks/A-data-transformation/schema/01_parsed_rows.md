# C1 parsed rows

This is the normative schema for `data/interim/parsed/rows.jsonl`. T1 writes one UTF-8, LF-terminated JSON object per `|`-separated NXT dialogue-act part. Keys are emitted in the order below.

| Field | Type | Meaning |
|---|---|---|
| `seg_id` | string | `{meeting}-{channel}_{src_start_ms:07d}_{src_end_ms:07d}`, with `_p1`, `_p2`, … for split acts. Milliseconds come from the source decimal string, never a float or derived time. |
| `meeting`, `agent` | string | Meeting and DA-file agent letter. |
| `channel`, `speaker_id` | string | Verbatim NXT `channel` and `participant`. |
| `src_start`, `src_end` | number | Parsed original DA element times in seconds. Rows with `time_invalid` retain the best available numeric placeholder and are never eligible for a clean build. |
| `start`, `end` | number | This part's derived span, constrained within the source span. |
| `timing_method` | string | `element`, `transcript_word_boundary`, `transcript_proportional`, or `source_proportional`. |
| `position` | integer | Zero-based meeting-local rank by `(src_start, src_end, agent, element_ordinal, part_index)`. |
| `source_da_id` | string | Verbatim NXT `nite:id`. |
| `element_ordinal` | integer | Zero-based document-order index inside its DA XML file. |
| `part_index`, `n_parts` | integer | Zero-based pipe-part index and the complete count. |
| `raw_type`, `original_type`, `part_type` | string or null | Original source values and the literal `|` part. T1 does not interpret `^`, `.`, or `:`. |
| `adjacency_raw`, `comment` | string or null | Verbatim NXT attributes; T2 owns their interpretation. |
| `text` | string | Non-empty `words[*].text` values joined with one space, preserving case, quotes, fillers, fragments, and non-`w` content. |
| `words` | list | Ordered objects: `{text, kind, start, end}`. `kind` is one of `w`, `disfmarker`, `vocalsound`, `nonvocalsound`, `comment`, `pause`; missing source times are `null`. |
| `quality_flags` | list of strings | Ordered, de-duplicated source-quality flags. |

## Timing methods

- `element`: an unsplit source act keeps its complete source span.
- `transcript_word_boundary`: a transcript pipe aligns to the ordered word range; a part boundary is the previous word's end, or next word's start when needed.
- `transcript_proportional`: a uniquely aligned pipe has no usable timing on at least one boundary, so boundaries are proportional to word counts within the source span. If every part is empty, the source span is divided evenly by part count.
- `source_proportional`: a valid source act has no word reference; its split parts divide the source span evenly by part count.

The alignment normalizer is only for comparison: it lowercases and ignores non-alphanumeric characters, while considering only lexical `w` elements. It never changes stored text. Punctuation-only source tokens remain with the preceding part. A boundary at the
start or end of the referenced range can therefore produce an empty part, which
is retained and reported.

## Pipe-boundary acceptance rule

For every source dialogue act whose type contains `|`, the parser must:

1. find a transcript record with the same meeting, channel, and source timestamps;
2. require the transcript to contain exactly the same number of parts as the source type;
3. resolve each pipe to one and only one ordered boundary in the referenced lexical `w` elements after normalization.

If any condition fails, the parser must not guess a split or copy transcript text into C1. It retains one inspection row per source part, adds `ref_unresolved`, records the source dialogue act and whether the transcript record is absent or lexically ambiguous in the validation report, and exits non-zero. Reviewed transcript-only corrections may be supplied through the tracked provenance correction table, but must not alter dialogue-act or word XML. Such output must not be frozen or used by downstream tasks until the source record or alignment rule is reviewed. Repeated part rows represent one source dialogue act and are reported once.

## Quality flags

Informational: `no_word_ref`, `type_fallback_original`, `word_time_missing`.
`word_time_missing` is emitted when any selected NXT word-like element has a
missing start or end time, including non-lexical markers; the report separates
row-level and element-level counts.

Error-class: `ref_unresolved`, `ref_malformed`, `time_invalid`, `type_unavailable`. An unresolved transcript pipe boundary is recorded as `ref_unresolved` because C1 has no separate pipe-error value. Error-class rows are retained for investigation and make the build exit non-zero.

## Serialization and validation

Rows sort by `(meeting, position)`. Serialization must be deterministic, use no NaN or Infinity, and depend on neither traversal order nor machine-specific values. `seg_id` uniqueness, contiguous positions, exact field order, and timing constraints are validated before a clean build is reported.
