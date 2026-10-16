import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ElementTree

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.parse_nxt import write_jsonl  # noqa: E402
from smi.data.nxt.ids import make_seg_id
from smi.data.nxt.ordering import assign_positions
from smi.data.nxt.parser import (
    derive_part_times,
    find_pipe_boundaries,
    parse_corpus,
    resolve_word_range,
    split_words,
)

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "icsi_core_nxt"


@pytest.mark.skipif(
    not (RAW_ROOT / "DialogueActs").is_dir(),
    reason="local ICSI NXT release is not available",
)
def test_generated_output_has_unique_ids(tmp_path: Path) -> None:
    rows, source_counts, _ = parse_corpus(RAW_ROOT)
    output_path = tmp_path / "rows.jsonl"

    write_jsonl(rows, output_path)
    serialized_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    segment_ids = [row["seg_id"] for row in serialized_rows]

    assert len(segment_ids) == source_counts["independent_pipe_parts"]
    assert len(segment_ids) == len(set(segment_ids))
    assert serialized_rows == rows


def test_comma_ids_and_non_word_endpoint_are_resolved() -> None:
    words = [
        {"text": "Hello", "kind": "w", "start": 1.0, "end": 1.1},
        {"text": "breath", "kind": "vocalsound", "start": None, "end": None},
    ]
    index = {"meeting.w.1171": 0, "meeting.vocalsound.2": 1}
    child = ElementTree.fromstring(
        '<nite:child xmlns:nite="http://nite.sourceforge.net/" '
        'href="words.xml#id(meeting.w.1,171)..id(meeting.vocalsound.2)"/>'
    )
    result, flags = resolve_word_range(child, words, index)
    assert [word["kind"] for word in result] == ["w", "vocalsound"]
    assert flags == ["word_time_missing"]


def test_missing_reference_is_informational() -> None:
    result, flags = resolve_word_range(None, [], {})
    assert result == []
    assert flags == ["no_word_ref"]


def test_pipe_alignment_keeps_punctuation_with_left_part() -> None:
    words = [
        {"text": "Oh", "kind": "w", "start": 0.0, "end": 0.1},
        {"text": ",", "kind": "w", "start": 0.1, "end": 0.1},
        {"text": "yes", "kind": "w", "start": 0.1, "end": 0.2},
    ]
    boundaries = find_pipe_boundaries(words, "oh | yes", 2)
    assert boundaries is not None
    split_parts = split_words(words, boundaries, 2)
    part_texts = []
    for part in split_parts:
        part_texts.append([word["text"] for word in part])
    assert part_texts == [
        ["Oh", ","],
        ["yes"],
    ]


def test_pipe_alignment_rejects_missing_transcript_boundary() -> None:
    words = [
        {"text": "hello", "kind": "w", "start": 0.0, "end": 0.1},
        {"text": "world", "kind": "w", "start": 0.1, "end": 0.2},
    ]
    assert find_pipe_boundaries(words, "missing | world", 2) is None


def test_pipe_alignment_rejects_wrong_part_count() -> None:
    words = [{"text": "hello", "kind": "w", "start": 0.0, "end": 0.1}]
    assert find_pipe_boundaries(words, "hello", 2) is None


def test_pipe_alignment_allows_leading_empty_part() -> None:
    words = [
        {"text": "and", "kind": "w", "start": 0.0, "end": 0.1},
        {"text": "listen", "kind": "w", "start": 0.1, "end": 0.2},
    ]
    boundaries = find_pipe_boundaries(words, "| and listen", 2)
    assert boundaries == [0]
    assert split_words(words, boundaries, 2) == [[], words]


def test_source_based_id_ignores_derived_part_timing() -> None:
    assert make_seg_id("Bdb001", "c1", "164.014", "165.974", 0, 2) == "Bdb001-c1_0164014_0165974_p1"


def test_source_proportional_timing_divides_empty_parts_evenly() -> None:
    timings, method = derive_part_times(10.0, 16.0, [[], [], []])
    assert method == "transcript_proportional"
    assert timings == [(10.0, 12.0), (12.0, 14.0), (14.0, 16.0)]


def test_split_timing_is_monotonic_when_word_boundaries_reverse() -> None:
    parts = [
        [{"text": "one", "kind": "w", "start": 0.0, "end": 5.0}],
        [{"text": "two", "kind": "w", "start": 3.0, "end": 4.0}],
        [{"text": "three", "kind": "w", "start": 4.0, "end": 6.0}],
    ]
    timings, method = derive_part_times(0.0, 6.0, parts)
    assert method == "transcript_word_boundary"
    assert timings == [(0.0, 5.0), (5.0, 5.0), (5.0, 6.0)]


def test_invalid_source_precision_is_not_converted_to_a_zero_id() -> None:
    with pytest.raises(ValueError, match="more than millisecond precision"):
        make_seg_id("M", "c0", "1.0005", "2.0", 0, 1)


def test_ordering_is_source_based_and_contiguous() -> None:
    rows = [
        {"meeting": "M", "src_start": 1.0, "src_end": 1.0, "agent": "B", "element_ordinal": 0, "part_index": 0},
        {"meeting": "M", "src_start": 1.0, "src_end": 1.0, "agent": "A", "element_ordinal": 1, "part_index": 1},
        {"meeting": "M", "src_start": 1.0, "src_end": 1.0, "agent": "A", "element_ordinal": 1, "part_index": 0},
    ]
    ordered = assign_positions(rows)
    assert [row["position"] for row in ordered] == [0, 1, 2]
    assert [row["agent"] + str(row["part_index"]) for row in ordered] == ["A0", "A1", "B0"]
