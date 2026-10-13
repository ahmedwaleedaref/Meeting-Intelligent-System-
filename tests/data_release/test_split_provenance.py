"""Task 3: the committed split provenance (coverage, folds, reproducibility, hashes)."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from smi.data.splits import (
    DEFAULT_FOLD_COUNT,
    DEFAULT_SPLIT_COUNTS,
    SPLIT_NAMES,
    assign_grouped_folds,
    assign_stratified_splits,
    attach_split_info,
    load_meeting_ids,
    write_split_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_FOLDER = PROJECT_ROOT / "tasks" / "A-data-transformation"
SPLIT_FILE = TASK_FOLDER / "provenance" / "split_v1.json"
METADATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "ICSI-metadata.xml",
    PROJECT_ROOT / "data" / "interim" / "splits" / "ICSI-metadata.xml",
)
METADATA_XML = next((path for path in METADATA_CANDIDATES if path.is_file()), METADATA_CANDIDATES[0])
RELEASES = sorted(path.name for path in (PROJECT_ROOT / "data").glob("v[0-9]*") if path.is_dir())


@pytest.fixture(scope="module")
def provenance() -> dict[str, Any]:
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def meetings(provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["meeting_id"]: record for record in provenance["meetings"]}


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_every_meeting_appears_exactly_once(provenance: dict[str, Any]) -> None:
    """No meeting id is repeated and each has exactly one split."""
    ids = [record["meeting_id"] for record in provenance["meetings"]]
    assert len(ids) == len(set(ids)) == sum(DEFAULT_SPLIT_COUNTS.values())
    assert {record["split"] for record in provenance["meetings"]} <= set(SPLIT_NAMES)


def test_split_sizes_match_the_documented_policy(provenance: dict[str, Any]) -> None:
    """The split sizes equal the recorded counts and the default 51/12/12 policy."""
    counts = Counter(record["split"] for record in provenance["meetings"])
    assert dict(counts) == DEFAULT_SPLIT_COUNTS == provenance["split_assignment"]["counts"]


def test_test_meetings_have_no_fold_and_all_others_have_one(provenance: dict[str, Any]) -> None:
    """Folds exist for train and val meetings only, each within [0, K)."""
    k = provenance["folds"]["k"]
    for record in provenance["meetings"]:
        if record["split"] == "test":
            assert record["fold"] is None
        else:
            assert isinstance(record["fold"], int) and 0 <= record["fold"] < k


def test_every_fold_is_used_and_folds_are_balanced(provenance: dict[str, Any]) -> None:
    """All K folds are used and their sizes differ by at most one meeting."""
    sizes = Counter(record["fold"] for record in provenance["meetings"] if record["fold"] is not None)
    assert set(sizes) == set(range(provenance["folds"]["k"]))
    assert max(sizes.values()) - min(sizes.values()) <= 1


def test_folds_record_k_grouping_and_method(provenance: dict[str, Any]) -> None:
    """K, the meeting grouping and the assignment method are written down."""
    folds = provenance["folds"]
    assert folds["k"] == DEFAULT_FOLD_COUNT and folds["grouping"] == "meeting"
    assert folds["method"] and "seed" in folds


def test_fold_assignment_is_the_explicit_round_robin_over_train_and_val(
    provenance: dict[str, Any], meetings: dict[str, dict[str, Any]]
) -> None:
    """Folds equal a sorted round-robin over the train and val meetings."""
    practice = [meeting for meeting, record in meetings.items() if record["split"] != "test"]
    expected = assign_grouped_folds(practice, provenance["folds"]["k"])
    assert {meeting: meetings[meeting]["fold"] for meeting in practice} == expected


def test_stratified_assignment_reproduces_from_the_meeting_ids(
    provenance: dict[str, Any], meetings: dict[str, dict[str, Any]]
) -> None:
    """Re-running the documented policy on the same ids gives the committed lists."""
    policy = provenance["split_assignment"]
    rebuilt = assign_stratified_splits(sorted(meetings), policy["counts"], seed=policy["seed"])
    committed = {split: sorted(m for m, r in meetings.items() if r["split"] == split) for split in SPLIT_NAMES}
    assert rebuilt == committed


def test_sources_and_reasons_are_recorded(provenance: dict[str, Any]) -> None:
    """The provenance names where the lists came from and why each meeting is where it is."""
    assert provenance["sources"]["split_lists"] and provenance["sources"]["meeting_universe"]
    assert all(record["reason"].strip() for record in provenance["meetings"])


def test_the_replacement_for_table_9_is_stated_openly(provenance: dict[str, Any]) -> None:
    """The file says it is not the lead's Table 9 list, and the README repeats it."""
    assert provenance["split_assignment"]["lead_table_9"] is False
    assert "Table 9" in (TASK_FOLDER / "README.md").read_text(encoding="utf-8")


@pytest.mark.skipif(not METADATA_XML.is_file(), reason="local ICSI-metadata.xml is not available")
def test_split_covers_exactly_the_meeting_universe(meetings: dict[str, dict[str, Any]]) -> None:
    """Every observation in ICSI-metadata.xml is in the split, and nothing else is."""
    assert set(load_meeting_ids(METADATA_XML)) == set(meetings)


@pytest.mark.parametrize("version", RELEASES)
def test_released_splits_json_matches_the_provenance(provenance: dict[str, Any], version: str) -> None:
    """Each release's splits.json has the same K, grouping and per-meeting assignments."""
    splits = json.loads((PROJECT_ROOT / "data" / version / "splits.json").read_text(encoding="utf-8"))
    assert (splits["k"], splits["grouping"]) == (provenance["folds"]["k"], "meeting")
    assert splits["meetings"] == {
        record["meeting_id"]: {"split": record["split"], "fold": record["fold"]} for record in provenance["meetings"]
    }


def test_attaching_split_info_gives_every_meeting_its_recorded_assignment(
    provenance: dict[str, Any], meetings: dict[str, dict[str, Any]]
) -> None:
    """attach_split_info reproduces the committed split and fold for every meeting."""
    rows = attach_split_info([{"meeting": meeting} for meeting in meetings], provenance)
    assert {row["meeting"]: (row["split"], row["fold"]) for row in rows} == {
        meeting: (record["split"], record["fold"]) for meeting, record in meetings.items()
    }


@pytest.mark.parametrize("version", RELEASES)
def test_recorded_split_hash_differs_from_the_committed_file_only_by_line_endings(version: str) -> None:
    """The hash stored in each release is the CRLF form of the committed split file."""
    recorded = json.loads((PROJECT_ROOT / "data" / version / "splits.json").read_text(encoding="utf-8"))["source_sha256"]
    lf = SPLIT_FILE.read_bytes().replace(b"\r\n", b"\n")
    assert recorded in {sha256_of(lf), sha256_of(lf.replace(b"\n", b"\r\n"))}


@pytest.mark.parametrize("version", RELEASES)
def test_released_split_hash_matches_the_committed_file_bytes(version: str) -> None:
    """source_sha256 in each release equals the SHA-256 of the committed split_v1.json."""
    recorded = json.loads((PROJECT_ROOT / "data" / version / "splits.json").read_text(encoding="utf-8"))["source_sha256"]
    assert recorded == sha256_of(SPLIT_FILE.read_bytes())


@pytest.mark.xfail(
    strict=True,
    reason="write_split_provenance calls write_text without newline='\\n', so Windows writes CRLF; "
    "every other writer in the repo sets newline='\\n'",
)
def test_split_provenance_is_written_with_explicit_lf_endings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provenance: dict[str, Any]
) -> None:
    """write_split_provenance must request LF line endings on every platform."""
    captured: dict[str, Any] = {}
    original = Path.write_text

    def spy(self: Path, data: str, encoding: str | None = None, errors: str | None = None, newline: str | None = None) -> int:
        captured["newline"] = newline
        return original(self, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", spy)
    write_split_provenance(provenance, tmp_path / "split.json")
    assert captured["newline"] == "\n"
