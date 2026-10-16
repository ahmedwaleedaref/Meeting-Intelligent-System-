"""Task 2: annotate_rows.py and clean_rows.py end to end on hand-written fixtures."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from annotate_fixtures import meeting_parsed
from scripts.annotate_rows import annotate_corpus
from scripts.annotate_rows import main as annotate_main
from scripts.clean_rows import fingerprint_problems
from scripts.clean_rows import main as clean_main
from smi.data.annotate.contract import check_contract
from smi.data.annotate.jsonl import read_parsed_rows, write_jsonl

M1 = [
    ("p", "1a", "s^cs"),
    ("z", "1b-1", "z"),
    ("r", "1b-2", "s^aa"),
    ("q", None, "s^arp^cs"),
]
M2 = [("r2", "1b", "s^aa"), ("p2", "1a", "s^co")]


@pytest.fixture
def parsed() -> list:
    return [*meeting_parsed(M1, "m1"), *meeting_parsed(M2, "m2")]


@pytest.fixture
def input_file(tmp_path: Path, parsed: list) -> Path:
    path = tmp_path / "parsed" / "rows.jsonl"
    write_jsonl(parsed, path)
    return path


def run_annotate(monkeypatch: pytest.MonkeyPatch, input_file: Path, root: Path) -> int:
    arguments = [
        "annotate_rows.py",
        "--input", str(input_file),
        "--output-dir", str(root / "annotated"),
        "--report", str(root / "annotation_report.md"),
        "--provenance", str(root / "c2.json"),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    return annotate_main()


def test_annotation_keeps_every_row_in_order_and_passes_the_contract(parsed: list) -> None:
    """Annotation returns the same rows in the same order and satisfies the contract."""
    annotated, links, _ = annotate_corpus(parsed)
    assert [row["seg_id"] for row in annotated] == [row["seg_id"] for row in parsed]
    assert check_contract(parsed, annotated, links) == []


def test_links_cover_exactly_the_proposals_of_every_meeting(parsed: list) -> None:
    """There is one link per proposal row across all meetings, in row order."""
    annotated, links, _ = annotate_corpus(parsed)
    assert [link["proposal_id"] for link in links] == [row["seg_id"] for row in annotated if row["is_layer2_proposal"]]
    assert [link["proposal_id"] for link in links] == ["p", "q", "p2"]


def test_pair_numbers_never_leak_between_meetings(parsed: list) -> None:
    """A response in one meeting never answers a proposal in another meeting."""
    _, links, _ = annotate_corpus(parsed)
    by_id = {link["proposal_id"]: link for link in links}
    assert by_id["p"]["responder_chains"] == [["z"], ["r"]]
    assert by_id["p2"]["responder_chains"] == []
    assert by_id["p2"]["status"] == "no_pair"


def test_audit_notes_are_collected_across_meetings(parsed: list) -> None:
    """Notes from every meeting are summed in one counter."""
    _, _, notes = annotate_corpus(parsed)
    assert notes["b_without_group"] == 1


def test_jsonl_round_trip_is_lossless_and_compact(tmp_path: Path, parsed: list) -> None:
    """Rows written as JSONL read back unchanged, as compact UTF-8 with LF endings."""
    path = tmp_path / "rows.jsonl"
    write_jsonl(parsed, path)
    data = path.read_bytes()
    assert read_parsed_rows(path) == parsed
    assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf") and data.endswith(b"\n")
    assert b'", "' not in data and b'": ' not in data


def test_annotation_run_writes_identical_files_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, input_file: Path
) -> None:
    """Two runs on the same input write byte-identical rows, links and report."""
    assert run_annotate(monkeypatch, input_file, tmp_path / "one") == 0
    assert run_annotate(monkeypatch, input_file, tmp_path / "two") == 0
    for name in ("annotated/rows.jsonl", "annotated/links.jsonl", "annotation_report.md"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_annotation_run_records_matching_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, input_file: Path
) -> None:
    """The provenance file holds the real hashes and counts of the files written."""
    assert run_annotate(monkeypatch, input_file, tmp_path) == 0
    record = json.loads((tmp_path / "c2.json").read_text(encoding="utf-8"))
    rows_path, links_path = tmp_path / "annotated" / "rows.jsonl", tmp_path / "annotated" / "links.jsonl"
    assert record["rows"]["sha256"] == hashlib.sha256(rows_path.read_bytes()).hexdigest()
    assert record["links"]["sha256"] == hashlib.sha256(links_path.read_bytes()).hexdigest()
    assert record["input"]["sha256"] == hashlib.sha256(input_file.read_bytes()).hexdigest()
    assert (record["input"]["row_count"], record["rows"]["row_count"], record["links"]["row_count"]) == (6, 6, 3)
    assert len(record["producing_commit"]) == 40


def test_contract_failure_writes_nothing_and_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parsed file that breaks the contract is rejected before any output is written."""
    duplicated = meeting_parsed([("p", "1a", "s^cs"), ("r", "1b", "s")])
    duplicated[1]["seg_id"] = duplicated[0]["seg_id"]
    input_file = tmp_path / "bad.jsonl"
    write_jsonl(duplicated, input_file)
    assert run_annotate(monkeypatch, input_file, tmp_path / "out") == 1
    assert "Contract problems found" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def run_clean(monkeypatch: pytest.MonkeyPatch, root: Path) -> int:
    arguments = [
        "clean_rows.py",
        "--input-dir", str(root / "annotated"),
        "--annotated-provenance", str(root / "c2.json"),
        "--output-dir", str(root / "cleaned"),
        "--report", str(root / "cleaning_report.md"),
        "--provenance", str(root / "c2_clean.json"),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    return clean_main()


def test_cleaning_run_drops_flagged_rows_and_records_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, input_file: Path
) -> None:
    """Cleaning removes the non-labeled row, prunes its link entry and records the counts."""
    assert run_annotate(monkeypatch, input_file, tmp_path) == 0
    assert run_clean(monkeypatch, tmp_path) == 0
    cleaned = read_parsed_rows(tmp_path / "cleaned" / "rows.jsonl")
    assert [row["seg_id"] for row in cleaned] == ["p", "r", "q", "r2", "p2"]
    links = [json.loads(line) for line in (tmp_path / "cleaned" / "links.jsonl").read_text(encoding="utf-8").splitlines()]
    assert links[0]["responder_chains"] == [["r"]]
    record = json.loads((tmp_path / "c2_clean.json").read_text(encoding="utf-8"))
    assert record["dropped"]["is_nonlabeled"] == 1 and record["dropped"]["unique"] == 1
    assert record["rows"]["row_count"] == 5


def test_cleaning_rejects_inputs_that_do_not_match_the_annotation_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, input_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rows file changed after annotation is rejected and nothing is written."""
    assert run_annotate(monkeypatch, input_file, tmp_path) == 0
    rows_path = tmp_path / "annotated" / "rows.jsonl"
    rows_path.write_bytes(rows_path.read_bytes() + b"\n")
    assert fingerprint_problems(tmp_path / "c2.json", rows_path, tmp_path / "annotated" / "links.jsonl") == [
        "rows do not match c2.json; re-run scripts/annotate_rows.py"
    ]
    assert run_clean(monkeypatch, tmp_path) == 1
    assert "Inputs rejected" in capsys.readouterr().err
    assert not (tmp_path / "cleaned").exists()
