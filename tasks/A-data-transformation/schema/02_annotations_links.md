# Schema 02 — Annotated rows, links and cleaning

Normative for Task 2's outputs. The input is C1 ([schema 01](01_parsed_rows.md)); the release built
from the cleaned output is in [schema 03](03_release.md). Background on the tag and adjacency
notation: [MRDA tags and adjacency pairs](../../../concepts/mrda-tags-and-adjacency-pairs.md).
Serialization is as in schema 01: UTF-8, LF, one JSON object per line, keys in the order listed,
no NaN, and the same input gives the same bytes. Page numbers refer to the labelling manual
(Dhillon et al., 2004).

## Annotated rows

`data/interim/annotated/rows.jsonl`. Every C1 field is passed through unchanged, with the same
rows in the same order. These fields are appended, in this order:

| Field | Type | Meaning |
|---|---|---|
| `label` | string | One of `cs, co, aa, bk, ar, cc, other` (the list lives only in `src/smi/labels.py`) |
| `label_bk_merged` | string | `label` computed again with a bare `b` tag counted as `bk` |
| `target_tags`, `is_multi_target` | list of strings, boolean | Every target class the act carries, in the order `cs, co, aa, bk, ar, cc`; true when there are two or more |
| `is_layer2_proposal` | boolean | `cs` or `co` is among `target_tags`; independent of `label` |
| `general_tags`, `disruption_markers` | list of strings | Analysis-only. Disruption markers are `%-` and `%--`; general tags are everything else that is not a target tag |
| `is_nonspeech` | boolean | The label, before any colon, is exactly `x`, and the row has no error flag |
| `is_nonlabeled` | boolean | The label, before any colon, is exactly `z`, and the row has no error flag |
| `is_empty` | boolean | `text` is empty and the row has no error flag |

### Reading the label string (`part_type`)

- Only the text before the first `:` is read. What follows is quoted material and belongs to the
  quoted speaker (p. 21).
- The text is split on `^` and `.`; empty pieces are dropped. Each piece is a tag.
- A target tag is `cs`, `co`, `aa`, `bk`, `ar`, `cc`, or `aap` (mapped to `aa`) or `arp` (mapped to
  `ar`). `target_tags` holds each mapped class once.
- `label` is `other` when `target_tags` is empty, the tag itself when there is one, and otherwise
  the highest of `cc > co > cs > ar > aa > bk`. The order is a convention (the manual does not rank
  the classes); the full set stays in `target_tags`, so it can be revisited.
- A bare `b` is not a target tag. It only affects `label_bk_merged`.
- `general_tags` keeps each remaining tag once, in order of appearance (`s`, `b`, `x`, `z`, `%`, …).
  `disruption_markers` keeps each of `%-` and `%--` once.
- An error flag in `quality_flags` (`ref_unresolved`, `ref_malformed`, `time_invalid`,
  `type_unavailable`) means the row is a parse problem, not data: it is never marked non-speech,
  non-labeled or empty.
- A label with an appended `x` (for example `s.x`) is an act with a disruption form (pp. 19–20)
  and is **not** non-speech. `z` is "deliberately unlabelled", not silent (pp. 17, 113–114).

## Links

`data/interim/annotated/links.jsonl`. Exactly one record per row with `is_layer2_proposal` true, in
row order.

| Field | Type | Meaning |
|---|---|---|
| `proposal_id` | string | The proposal's `seg_id` |
| `responder_chains` | list of lists of strings | One chain per replying speaker number, each a list of `seg_id`s in order, ordered by the position of each chain's first row; empty when there is no reply |
| `has_response` | boolean | `responder_chains` is not empty |
| `status` | string | `ok`, `no_pair`, `ambiguous` or `malformed` (below) |
| `adjacency_tokens` | list of strings | The raw adjacency string split on `.`; empty when the act has none |
| `pair_group` | string or null | `{meeting}:{number}{underscores}@{position}` of the first pair the act opens or extends (the position is that of the pair's first row), or null |

### Reading the adjacency string (`adjacency_raw`)

- A piece matches `<number><underscores><a|b>[-<speaker>]<pluses>`, for example `5a`, `5b`,
  `5b-1`, `14b++`, `8_b`. Any other piece is unreadable.
- Tokens belong to the whole act (p. 25): every part of a split act joins the same pair, so a
  reply adds all of its parts to a chain.
- A pair is keyed by its number and its underscores. A plain `a` opens a pair. An `a` with `+`, or
  with a speaker number above 1, extends the open pair with the same key (a proposal-side
  continuation); with no open pair it creates a pair marked ambiguous.
- A `b` adds the act to the chain of its speaker number (chain 0 when it has none) in the open pair
  with the same key. With no open pair the reply is not attached. A bare `b+` extends the only
  chain; with several chains it is not attached and the pair is marked ambiguous.
- **Restart.** When a plain `a` numbered 1 appears after higher numbers have been used, every open
  pair is closed. The numbering restarts this way within a meeting, so a number cannot identify a
  pair for the whole meeting.
- Rows are read in `position` order, never by derived times.

### Status

- `malformed`: at least one piece of the adjacency string is unreadable (an empty string counts).
- `ambiguous`: a pair this act opened or extended is marked ambiguous (see above). Never converted to `no_pair`.
- `ok`: the act opened or extended a pair and at least one reply was attached.
- `no_pair`: none of the above. This covers an act with no adjacency string, an act that only
  replies to another pair, and a pair that nobody answered.
- Precedence: `malformed`, then `ambiguous`, then `ok` or `no_pair`.

## Cleaning

`data/interim/cleaned/rows.jsonl` and `links.jsonl`, written by a separate step.

- A row is removed when `is_nonspeech`, `is_nonlabeled` or `is_empty` is true.
- A kept row has the C1 fields plus `label`, `label_bk_merged`, `target_tags`, `is_multi_target`,
  `is_layer2_proposal`, `general_tags` and `disruption_markers` (30 fields). The three flags are
  removed, since they would all be false.
- `seg_id` and `position` are never changed, so `position` has gaps where rows were removed.
- Links: the record of a removed proposal is removed (none is on the corpus); removed `seg_id`s
  leave the chains, empty chains are dropped, `has_response` is recomputed, and an `ok` record left
  with no chain becomes `no_pair`. `adjacency_tokens` and `pair_group` are unchanged.

## Checks before anything is written

Annotation stops with a list of problems, writing nothing, unless all of these hold: the row count
and every C1 field are unchanged; keys are in schema order; `seg_id` is unique and rows are sorted
by `(meeting, position)`; labels are in `LABELS`; `target_tags` are distinct and ordered, `label`
follows them, and the multi-target and proposal flags agree with them; general tags hold no target
tag or marker; a row is not both non-speech and non-labeled; `is_empty` implies empty text; an
error-flagged row carries none of the three flags; links are exactly one per proposal in row
order, every chain member is a real row and appears once, chains are ordered by first position,
`has_response` agrees with the chains, `no_pair` has no reply, `ok` has a `pair_group`, and
`adjacency_tokens` match the row. Cleaning checks that its output equals the annotated rows minus
the removed ones.
