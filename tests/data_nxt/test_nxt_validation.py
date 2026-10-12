"""Task 1: C1 contract checks, the generated validation report and JSONL serialization."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nxt_fixtures import act, build_single_act_corpus, build_standard_corpus
from scripts.parse_nxt import sha256, write_jsonl
from smi.data.nxt.parser import parse_corpus
from smi.data.nxt.validation import validate_rows, write_validation_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_FOLDER = PROJECT_ROOT / "tasks" / "A-data-transformation"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "icsi_core_nxt"
C1_ROWS = PROJECT_ROOT / "data" / "interim" / "parsed" / "rows.jsonl"


@pytest.fixture
def rows(standard_parse: Any) -> list[dict[str, Any]]:
    return copy.deepcopy(standard_parse[0])


def test_clean_rows_have_no_contract_problem(rows: list[dict[str, Any]]) -> None:
    """Rows straight from the parser satisfy every C1 contract check."""
    assert validate_rows(rows) == []


def test_duplicate_seg_id_is_reported(rows: list[dict[str, Any]]) -> None:
    """Two rows sharing a seg_id fail the uniqueness check."""
    rows[1]["seg_id"] = rows[0]["seg_id"]
    assert "seg_id values are not unique" in validate_rows(rows)


def test_field_order_change_is_reported(rows: list[dict[str, Any]]) -> None:
    """A row whose keys are not in schema order is reported."""
    rows[0] = dict(reversed(list(rows[0].items())))
    assert any("field order" in problem for problem in validate_rows(rows))


def test_non_contiguous_positions_are_reported(rows: list[dict[str, Any]]) -> None:
    """A gap in a meeting's positions is reported."""
    rows[3]["position"] += 5
    assert any("positions are not contiguous" in problem for problem in validate_rows(rows))


def test_derived_timing_outside_the_source_span_is_reported(rows: list[dict[str, Any]]) -> None:
    """A derived span that leaves the source span fails the timing check."""
    rows[0]["end"] = rows[0]["src_end"] + 1.0
    assert any("outside source timing" in problem for problem in validate_rows(rows))


def test_overlapping_split_parts_are_reported(rows: list[dict[str, Any]]) -> None:
    """Parts of one act whose spans overlap fail the split-timing check."""
    parts = [row for row in rows if row["source_da_id"] == "Mtg001.A.dialogueact2"]
    parts[0]["end"] = parts[1]["start"] + 0.2
    assert any("split-part timings overlap" in problem for problem in validate_rows(rows))


def test_validation_report_states_counts_flags_and_reconciliation(
    tmp_path: Path, standard_parse: Any
) -> None:
    """The report states element, row and part counts, the reconciliation and flag totals."""
    rows, counts, problems = standard_parse
    path = tmp_path / "reports" / "parse_validation.md"
    write_validation_report(path, rows, counts, problems, validate_rows(rows))
    text = path.read_text(encoding="utf-8")
    assert "- Dialogue-act elements: 10" in text
    assert "- Generated C1 rows: 13" in text
    assert "- Reconciliation: matches." in text
    assert "- 1 part(s): 8" in text and "- 2 part(s): 2" in text and "- 3 part(s): 3" in text
    assert "- `no_word_ref`: 1" in text
    assert "- `type_fallback_original`: 1" in text
    assert "- Error-class source dialogue acts: 0" in text
    assert "Table 5 comparison: pending" in text
    assert "passed" in text and "FAILED" not in text


def test_validation_report_lists_error_rows_and_contract_failures(tmp_path: Path) -> None:
    """Error-class rows and contract failures are listed with their source ids."""
    root = build_single_act_corpus(tmp_path / "raw", act("Mtg001.A.dialogueact1", "1.0", "2.0", None))
    rows, counts, problems = parse_corpus(root)
    path = tmp_path / "report.md"
    write_validation_report(path, rows, counts, problems, ["synthetic contract failure"])
    text = path.read_text(encoding="utf-8")
    assert "`Mtg001.A.dialogueact1`: type_unavailable" in text
    assert "Error-class source dialogue acts: 1" in text
    assert "synthetic contract failure" in text and "FAILED" in text


def test_validation_report_flags_a_count_mismatch(tmp_path: Path, standard_parse: Any) -> None:
    """A row count that differs from the independent part count is reported as DIFFERS."""
    rows, counts, problems = standard_parse
    counts = counts.copy()
    counts["independent_pipe_parts"] += 1
    path = tmp_path / "report.md"
    write_validation_report(path, rows, counts, problems, [])
    assert "Reconciliation: DIFFERS." in path.read_text(encoding="utf-8")


def test_validation_report_is_reproducible_and_has_no_timestamp(
    tmp_path: Path, standard_parse: Any
) -> None:
    """Writing the report twice gives identical bytes with no date or path in it."""
    rows, counts, problems = standard_parse
    first, second = tmp_path / "one.md", tmp_path / "two.md"
    write_validation_report(first, rows, counts, problems, [])
    write_validation_report(second, rows, counts, problems, [])
    assert first.read_bytes() == second.read_bytes()
    assert str(tmp_path) not in first.read_text(encoding="utf-8")


def test_jsonl_is_utf8_lf_compact_and_keeps_key_order(tmp_path: Path, rows: list[dict[str, Any]]) -> None:
    """The file is UTF-8 without BOM, LF-terminated, compact, non-ASCII and key-ordered."""
    path = tmp_path / "rows.jsonl"
    write_jsonl(rows, path)
    data = path.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data and data.endswith(b"\n")
    assert "café".encode("utf-8") in data
    lines = data.decode("utf-8").splitlines()
    assert len(lines) == len(rows)
    assert ", " not in lines[0] and ": " not in lines[0]
    assert [list(json.loads(line)) for line in lines] == [list(row) for row in rows]
    assert [json.loads(line) for line in lines] == rows


def test_jsonl_rejects_non_finite_numbers(tmp_path: Path) -> None:
    """NaN and Infinity cannot be written into the artifact."""
    with pytest.raises(ValueError):
        write_jsonl([{"value": float("nan")}], tmp_path / "bad.jsonl")


def test_two_builds_write_byte_identical_files(tmp_path: Path) -> None:
    """Two independent parses of the same corpus serialize to identical bytes."""
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    write_jsonl(parse_corpus(build_standard_corpus(tmp_path / "one"))[0], first)
    write_jsonl(parse_corpus(build_standard_corpus(tmp_path / "two"))[0], second)
    assert first.read_bytes() == second.read_bytes()
    assert sha256(first) == hashlib.sha256(first.read_bytes()).hexdigest()


def test_source_archive_provenance_records_name_release_and_hash() -> None:
    """The archive provenance names the archive, its release and a SHA-256."""
    record = json.loads((TASK_FOLDER / "provenance" / "source_archive.json").read_text(encoding="utf-8"))
    assert record["archive_name"] == "ICSI_core_NXT.zip"
    assert record["release"]
    assert len(record["sha256"]) == 64 and record["sha256"] == record["sha256"].lower()


def test_c1_provenance_records_hash_count_and_commit() -> None:
    """The C1 fingerprint file holds a hash, a row count and the producing commit."""
    record = json.loads((TASK_FOLDER / "provenance" / "c1.json").read_text(encoding="utf-8"))
    assert len(record["sha256"]) == 64
    assert isinstance(record["row_count"], int) and record["row_count"] > 0
    assert len(record["producing_commit"]) == 40


def test_committed_parse_report_reconciles_with_the_c1_fingerprint() -> None:
    """The committed report shows a matching reconciliation and the fingerprinted row count."""
    record = json.loads((TASK_FOLDER / "provenance" / "c1.json").read_text(encoding="utf-8"))
    text = (TASK_FOLDER / "reports" / "parse_validation.md").read_text(encoding="utf-8")
    assert f"- Generated C1 rows: {record['row_count']}" in text
    assert "- Reconciliation: matches." in text
    assert "- Error-class source dialogue acts: 0" in text
    assert "FAILED" not in text


@pytest.mark.skipif(not C1_ROWS.is_file(), reason="local C1 artifact is not available")
def test_local_c1_file_matches_the_committed_fingerprint() -> None:
    """A locally generated C1 file hashes to the committed fingerprint."""
    record = json.loads((TASK_FOLDER / "provenance" / "c1.json").read_text(encoding="utf-8"))
    assert sha256(C1_ROWS) == record["sha256"]


@pytest.mark.skipif(not (RAW_ROOT / "DialogueActs").is_dir(), reason="local ICSI NXT release is not available")
def test_real_corpus_parses_deterministically_without_error_rows() -> None:
    """The real corpus parses twice to identical rows with no error-class row."""
    corrections_path = TASK_FOLDER / "provenance" / "transcript_corrections.json"
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    first, counts, problems = parse_corpus(RAW_ROOT, corrections)
    assert parse_corpus(RAW_ROOT, corrections)[0] == first
    assert len(first) == counts["independent_pipe_parts"]
    assert problems == []
    assert validate_rows(first) == []
