"""Task 1: parse_corpus on a hand-written NXT corpus (rows, timing, flags, ordering)."""

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ElementTree

import pytest

from nxt_fixtures import (
    act,
    build_single_act_corpus,
    build_standard_corpus,
    word,
    words_href,
    write_corpus,
)
from smi.data.nxt.parser import (
    has_fatal_errors,
    load_transcripts,
    parse_corpus,
    parse_time,
    transcript_key,
    word_text,
)
from smi.data.nxt.validation import C1_FIELDS, ERROR_FLAGS, validate_rows

Rows = dict[str, dict[str, Any]]


def test_unsplit_act_keeps_source_span_and_element_method(standard_rows: Rows) -> None:
    """An unsplit act keeps its source span, element timing and a plain seg_id."""
    row = standard_rows["Mtg001-c1_0001000_0001400"]
    assert (row["start"], row["end"]) == (row["src_start"], row["src_end"]) == (1.0, 1.4)
    assert (row["timing_method"], row["n_parts"], row["part_index"]) == ("element", 1, 0)
    assert row["text"] == "Hello there"
    assert row["quality_flags"] == []


def test_two_part_split_uses_suffixed_ids_and_word_boundary_timing(standard_rows: Rows) -> None:
    """A two-part act gets _p1/_p2 ids and a boundary at the previous word's end."""
    first = standard_rows["Mtg001-c1_0002000_0003000_p1"]
    second = standard_rows["Mtg001-c1_0002000_0003000_p2"]
    assert (first["text"], second["text"]) == ("yes", "okay")
    assert (first["start"], first["end"]) == (2.0, 2.3)
    assert (second["start"], second["end"]) == (2.3, 3.0)
    assert first["timing_method"] == second["timing_method"] == "transcript_word_boundary"
    assert (first["part_type"], second["part_type"]) == ("s^aa", "s^bk")
    assert first["raw_type"] == second["raw_type"] == "s^aa|s^bk"


def test_three_part_split_is_ordered_and_non_overlapping(standard_rows: Rows) -> None:
    """A three-part act yields three ordered parts whose spans never overlap."""
    parts = [standard_rows[f"Mtg001-c1_0004000_0007000_p{number}"] for number in (1, 2, 3)]
    assert [part["text"] for part in parts] == ["one", "two", "three"]
    assert [(part["start"], part["end"]) for part in parts] == [(4.0, 4.5), (4.5, 5.5), (5.5, 7.0)]
    assert {part["n_parts"] for part in parts} == {3}
    assert [part["part_type"] for part in parts] == ["s", "fg", "s^cs"]


def test_every_part_repeats_adjacency_and_comment_verbatim(standard_rows: Rows) -> None:
    """Adjacency and comment are copied unchanged onto every part of a split act."""
    assert standard_rows["Mtg001-c1_0001000_0001400"]["adjacency_raw"] == "1a"
    assert standard_rows["Mtg001-c1_0001000_0001400"]["comment"] == "note"
    for suffix in ("p1", "p2"):
        row = standard_rows[f"Mtg001-c1_0002000_0003000_{suffix}"]
        assert row["adjacency_raw"] == "1b"
        assert row["comment"] is None


def test_act_without_word_reference_is_empty_and_informational(standard_rows: Rows) -> None:
    """An act with no word reference keeps empty words and only the no_word_ref flag."""
    row = standard_rows["Mtg001-c1_0008000_0008500"]
    assert (row["text"], row["words"]) == ("", [])
    assert row["quality_flags"] == ["no_word_ref"]


def test_missing_type_falls_back_to_original_type_with_flag(standard_rows: Rows) -> None:
    """A missing type uses original-type for the part and records type_fallback_original."""
    row = standard_rows["Mtg001-c1_0009000_0009500"]
    assert row["raw_type"] is None
    assert (row["original_type"], row["part_type"]) == ("s^arp", "s^arp")
    assert row["quality_flags"] == ["type_fallback_original"]


def test_words_without_times_keep_null_times_and_flag(standard_rows: Rows) -> None:
    """Words with no source times stay null and raise word_time_missing."""
    row = standard_rows["Mtg001-c1_0010000_0010000"]
    assert row["words"] == [{"text": "um", "kind": "w", "start": None, "end": None}]
    assert row["quality_flags"] == ["word_time_missing"]
    assert row["start"] == row["end"] == 10.0


def test_comma_formatted_ids_resolve_in_a_full_parse(standard_rows: Rows) -> None:
    """A word id containing a thousands comma resolves while the stored text is untouched."""
    row = standard_rows["Mtg001-c1_0011000_0011400"]
    assert row["text"] == "comma"
    assert row["quality_flags"] == []


def test_range_may_end_on_a_non_word_element(standard_rows: Rows) -> None:
    """A range ending on a vocalsound keeps both element kinds and uses its description."""
    row = standard_rows["Mtg001-c1_0013000_0013400"]
    assert [item["kind"] for item in row["words"]] == ["w", "vocalsound"]
    assert row["text"] == "so laugh"


def test_cross_agent_ties_break_by_agent_letter(standard_parse: Any) -> None:
    """Two acts with identical source spans on different agents order by agent letter."""
    rows = standard_parse[0]
    tied = [row for row in rows if (row["src_start"], row["src_end"]) == (1.0, 1.4)]
    assert [(row["agent"], row["position"]) for row in tied] == [("A", 0), ("B", 1)]


def test_positions_are_contiguous_and_follow_the_source_order(standard_parse: Any) -> None:
    """Positions run 0..n-1 and every part of an act stays in part_index order."""
    rows = standard_parse[0]
    assert [row["position"] for row in rows] == list(range(len(rows)))
    keys = [(row["src_start"], row["src_end"], row["agent"], row["element_ordinal"], row["part_index"]) for row in rows]
    assert keys == sorted(keys)


def test_zero_duration_act_is_ordered_and_valid(standard_rows: Rows) -> None:
    """A zero-duration act keeps equal start and end and a unique id and position."""
    row = standard_rows["Mtg001-c1_0010000_0010000"]
    assert row["src_start"] == row["src_end"] == 10.0
    assert row["position"] == 9


def test_times_compare_numerically_not_as_strings(tmp_path: Path) -> None:
    """A 9.0 s act sorts before a 10.0 s act even though '10.0' sorts first as text."""
    acts = [act("Mtg001.A.dialogueact1", "10.0", "10.5"), act("Mtg001.A.dialogueact2", "9.0", "9.5")]
    root = write_corpus(tmp_path / "raw", words={"A": []}, acts={"A": acts})
    rows = parse_corpus(root)[0]
    assert [row["src_start"] for row in rows] == [9.0, 10.0]
    assert [row["position"] for row in rows] == [0, 1]


def test_clean_corpus_reconciles_with_independent_pipe_count(standard_parse: Any) -> None:
    """Row count equals the independent count of pipe-separated source type parts."""
    rows, counts, problems = standard_parse
    assert len(rows) == counts["independent_pipe_parts"] == 13
    assert counts["dialogue_act_elements"] == 10
    assert (counts["dialogue_act_files"], counts["meetings"]) == (2, 1)
    assert problems == []


def test_clean_corpus_passes_the_c1_contract(standard_parse: Any) -> None:
    """The fixture corpus raises no contract problem and no fatal error."""
    rows, _, problems = standard_parse
    assert validate_rows(rows) == []
    assert has_fatal_errors(rows, problems, []) is False


def test_seg_ids_are_unique_and_unchanged_when_word_times_change(tmp_path: Path) -> None:
    """Shifting word timings moves derived spans but never a seg_id or a position."""
    original = parse_corpus(build_standard_corpus(tmp_path / "one"))[0]
    shifted = parse_corpus(build_standard_corpus(tmp_path / "two", shift="0.1"))[0]
    original_ids = [row["seg_id"] for row in original]
    assert len(original_ids) == len(set(original_ids))
    assert original_ids == [row["seg_id"] for row in shifted]
    assert [row["position"] for row in original] == [row["position"] for row in shifted]
    changed = [pair for pair in zip(original, shifted) if pair[0]["end"] != pair[1]["end"]]
    assert changed


def test_every_row_has_every_c1_field_in_schema_order_with_types(standard_parse: Any) -> None:
    """Each row has all C1 fields in schema order with the documented types."""
    rows = standard_parse[0]
    expected_types = {
        "seg_id": str, "meeting": str, "agent": str, "channel": str, "speaker_id": str,
        "src_start": float, "src_end": float, "start": float, "end": float,
        "timing_method": str, "position": int, "source_da_id": str, "element_ordinal": int,
        "part_index": int, "n_parts": int, "text": str, "words": list, "quality_flags": list,
    }
    for row in rows:
        assert tuple(row) == C1_FIELDS
        for field, expected in expected_types.items():
            assert type(row[field]) is expected, field
        for field in ("raw_type", "original_type", "part_type", "adjacency_raw", "comment"):
            assert row[field] is None or isinstance(row[field], str)


def test_parse_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Two parses of the same corpus return identical rows and counts."""
    root = build_standard_corpus(tmp_path / "raw")
    first = parse_corpus(root)
    second = parse_corpus(root)
    assert first[0] == second[0]
    assert first[1] == second[1]


def test_non_ascii_source_text_survives_parsing(standard_rows: Rows) -> None:
    """Latin-1 source text is decoded and kept exactly in the row text."""
    assert standard_rows["Mtg001-c1_0012000_0012500"]["text"] == "café"


def test_missing_transcripts_directory_is_rejected(tmp_path: Path) -> None:
    """A raw root without a transcripts directory fails loudly instead of guessing splits."""
    root = build_single_act_corpus(tmp_path / "raw", act("Mtg001.A.dialogueact1", "1.0", "2.0"))
    (root / "transcripts" / "Mtg001.trans").unlink()
    (root / "transcripts").rmdir()
    with pytest.raises(FileNotFoundError, match="transcripts"):
        parse_corpus(root)


def test_missing_dialogue_act_directory_is_rejected(tmp_path: Path) -> None:
    """A raw root without DialogueActs fails loudly."""
    (tmp_path / "Words").mkdir()
    (tmp_path / "transcripts").mkdir()
    with pytest.raises(FileNotFoundError, match="DialogueActs"):
        parse_corpus(tmp_path)


ERROR_CASES = {
    "ref_unresolved": act("Mtg001.A.dialogueact1", "1.0", "2.0", "s", href=words_href("Mtg001.w.999")),
    "ref_malformed": act("Mtg001.A.dialogueact1", "1.0", "2.0", "s", href="not-a-reference"),
    "time_invalid": act("Mtg001.A.dialogueact1", "5.0", "4.0", "s"),
    "type_unavailable": act("Mtg001.A.dialogueact1", "1.0", "2.0", None),
}


@pytest.mark.parametrize("flag", sorted(ERROR_CASES))
def test_error_flag_keeps_the_row_and_blocks_a_clean_build(tmp_path: Path, flag: str) -> None:
    """Each error-class source problem keeps its row, sets its flag and is fatal."""
    words = [word("Mtg001.w.1", "hello", "1.0", "1.5")]
    root = build_single_act_corpus(tmp_path / "raw", ERROR_CASES[flag], words)
    rows, counts, problems = parse_corpus(root)
    assert len(rows) == counts["independent_pipe_parts"] == 1
    assert flag in rows[0]["quality_flags"]
    assert rows[0]["text"] == "" and rows[0]["words"] == []
    assert has_fatal_errors(rows, problems, validate_rows(rows)) is True


def test_reversed_source_times_are_flagged_not_repaired(tmp_path: Path) -> None:
    """A start after its end is flagged and the span is collapsed instead of reordered."""
    root = build_single_act_corpus(tmp_path / "raw", ERROR_CASES["time_invalid"])
    row = parse_corpus(root)[0][0]
    assert "time_invalid" in row["quality_flags"]
    assert row["src_start"] <= row["src_end"]


def test_unaligned_pipe_keeps_one_row_per_part_and_reports_the_act(tmp_path: Path) -> None:
    """A pipe act with no transcript keeps empty parts, flags them and lists the problem."""
    piped = act("Mtg001.A.dialogueact1", "1.0", "3.0", "s|s^aa", href=words_href("Mtg001.w.1", "Mtg001.w.2"))
    words = [word("Mtg001.w.1", "a", "1.0", "1.5"), word("Mtg001.w.2", "b", "2.0", "2.5")]
    rows, counts, problems = parse_corpus(build_single_act_corpus(tmp_path / "raw", piped, words))
    assert len(rows) == counts["independent_pipe_parts"] == 2
    assert all("ref_unresolved" in row["quality_flags"] for row in rows)
    assert all(row["text"] == "" for row in rows)
    assert [problem["source_da_id"] for problem in problems] == ["Mtg001.A.dialogueact1"]
    assert problems[0]["reason"] == "no pipe-preserving transcript record"


def test_ambiguous_transcript_wording_is_reported_as_unaligned(tmp_path: Path) -> None:
    """A transcript whose wording does not match the words is reported, never guessed."""
    piped = act("Mtg001.A.dialogueact1", "1.0", "3.0", "s|s^aa", href=words_href("Mtg001.w.1", "Mtg001.w.2"))
    words = [word("Mtg001.w.1", "a", "1.0", "1.5"), word("Mtg001.w.2", "b", "2.0", "2.5")]
    transcripts = ["Mtg001-c1_0001000_0003000,zzz | yyy"]
    _, _, problems = parse_corpus(build_single_act_corpus(tmp_path / "raw", piped, words, transcripts))
    assert problems[0]["reason"] == "transcript wording did not align uniquely"


def test_pipe_act_without_word_reference_splits_the_source_span_evenly(tmp_path: Path) -> None:
    """A pipe act with no word reference divides its span evenly and stays informational."""
    piped = act("Mtg001.A.dialogueact1", "2.0", "4.0", "s|s^aa")
    rows, _, problems = parse_corpus(build_single_act_corpus(tmp_path / "raw", piped))
    assert problems == []
    assert [(row["start"], row["end"]) for row in rows] == [(2.0, 3.0), (3.0, 4.0)]
    assert {row["timing_method"] for row in rows} == {"source_proportional"}
    assert all(row["quality_flags"] == ["no_word_ref"] for row in rows)


def test_error_flags_match_the_documented_vocabulary() -> None:
    """The error-class vocabulary is exactly the four flags in the schema."""
    assert ERROR_FLAGS == {"ref_unresolved", "ref_malformed", "time_invalid", "type_unavailable"}


def test_parse_time_rejects_blank_and_unreadable_values() -> None:
    """parse_time returns None for missing, blank or non-numeric values."""
    assert parse_time(None) is None
    assert parse_time("   ") is None
    assert parse_time("abc") is None
    assert parse_time("1.25") == 1.25


def test_word_text_uses_description_for_non_verbal_elements() -> None:
    """Non-verbal elements without text use their description attribute."""
    assert word_text(ElementTree.fromstring('<vocalsound description="cough"/>')) == "cough"
    assert word_text(ElementTree.fromstring("<w>  hi </w>")) == "hi"


def test_transcript_key_maps_the_two_c_channel_spelling() -> None:
    """A 2c0 channel maps to the transcript spelling c0 and times are zero-padded."""
    assert transcript_key("Bmr001", "2c0", "1.5", "2") == "Bmr001-c0_0001500_0002000"
    assert transcript_key("Bmr001", "C1", "0", "0.001") == "Bmr001-c1_0000000_0000001"


def test_load_transcripts_keeps_pipe_renderings_and_applies_corrections(tmp_path: Path) -> None:
    """Only renderings containing a pipe are kept, and corrections replace records."""
    (tmp_path / "a.trans").write_text(
        "id1,plain text,with | pipe\nid2,no pipes here\nid3,x | y,x | z\n", encoding="utf-8"
    )
    records = load_transcripts(tmp_path, {"id2": ["fixed | text"]})
    assert records["id1"] == ["with | pipe"]
    assert records["id2"] == ["fixed | text"]
    assert records["id3"] == ["x | y", "x | z"]
