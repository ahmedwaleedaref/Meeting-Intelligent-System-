# ICSI Core NXT annotations

The ICSI Core NXT input has `DialogueActs`, `Words`, and `transcripts` directories. A dialogue-act XML file supplies act metadata—times, speaker, channel, MRDA type, and optional adjacency data. Its `nite:child` element points into a same-agent words XML file using an inclusive NITE ID range.

The words files include ordinary lexical `w` elements and source content such as `disfmarker`, `vocalsound`, `nonvocalsound`, `comment`, and `pause`. T1 keeps every element in a referenced range, including a non-`w` endpoint. NXT remains the canonical source of these words and metadata.

Some NXT types contain `|`, which means that one source range represents several dialogue-act portions. The meeting transcript provides an auxiliary, pipe-preserving text record. T1 aligns its literal pipe to the NXT word range only to determine a split boundary; it does not replace the NXT annotation or normalize C1 text. A non-unique alignment is surfaced as an error rather than guessed.

The acceptance rule is strict: the transcript must be found under the exact source meeting, channel, and millisecond timestamps; its pipe count must equal the number of source parts; and every pipe must map to one ordered lexical boundary after comparison-only normalization. Reviewed transcript-only exceptions are stored in `tasks/A-data-transformation/provenance/transcript_corrections.json`; they never modify NXT dialogue-act or word data. A missing record may be reconstructed from a reviewed canonical NXT lexical range only when the reconstruction is explicitly documented as synthetic rather than recovered transcript text. A failed check produces inspection output and a non-zero build, so unresolved source records cannot silently enter frozen gold.

T1 deliberately does not interpret MRDA tag syntax beyond the literal pipe. In particular, tag classes, `^`, `.`, `:`, and adjacency relationships are T2 concerns.
