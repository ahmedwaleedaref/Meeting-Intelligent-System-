"""Task 3: load_release file checks and validate_release_content rejections."""

import copy
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from release_fixtures import build_release, read_release_parts, rewrite_checksums
from smi.data.release import RELEASE_FILES, load_release, validate_release_content

Parts = tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]


@pytest.fixture
def built(tmp_path: Path) -> Path:
    return build_release(tmp_path)[0]


@pytest.fixture
def parts(built: Path) -> Parts:
    return read_release_parts(built)


def _row(index: int, **changes: Any) -> Callable[[Parts], None]:
    return lambda parts: parts[0][index].update(changes)


def _link(index: int, **changes: Any) -> Callable[[Parts], None]:
    return lambda parts: parts[1][index].update(changes)


def _delete_row_field(parts: Parts) -> None:
    del parts[0][0]["label"]


def _swap_rows(parts: Parts) -> None:
    parts[0][0], parts[0][1] = parts[0][1], parts[0][0]


def _split(**changes: Any) -> Callable[[Parts], None]:
    return lambda parts: parts[2].update(changes)


def _declared_meeting(meeting: str, **assignment: Any) -> Callable[[Parts], None]:
    def mutate(parts: Parts) -> None:
        parts[2]["meetings"][meeting] = {**parts[2]["meetings"].get(meeting, {}), **assignment}

    return mutate


def _duplicate_link(parts: Parts) -> None:
    parts[1].append(copy.deepcopy(parts[1][0]))


def _drop_link(parts: Parts) -> None:
    parts[1].pop(0)


def _delete_link_field(parts: Parts) -> None:
    del parts[1][0]["status"]


ROW_CASES = [
    ("fold count", _split(k=1), "splits.k must be an integer of at least 2"),
    ("grouping", _split(grouping="series"), "splits.grouping must be 'meeting'"),
    ("meetings type", _split(meetings=[]), "splits.meetings must map"),
    ("source hash", _split(source_sha256="ABC"), "lowercase SHA-256"),
    ("missing field", _delete_row_field, "missing mandatory fields"),
    ("empty seg_id", _row(0, seg_id=""), "non-empty string"),
    ("negative position", _row(0, position=-1), "invalid position"),
    ("duplicate position", _row(1, position=0), "duplicate position in meeting"),
    ("unsorted rows", _swap_rows, r"sorted by \(meeting, position\)"),
    ("timing outside source", _row(0, end=99.0), "inconsistent source and dialogue-act timings"),
    ("nan time", _row(0, start=float("nan")), "invalid start"),
    ("boolean time", _row(0, start=True), "invalid start"),
    ("part index", _row(0, part_index=1), "inconsistent part_index and n_parts"),
    ("words shape", _row(0, words=[{"text": "a"}]), "invalid words"),
    ("label", _row(0, label="zz"), "outside LABELS"),
    ("tag order", _row(2, target_tags=["cc", "cs"]), "canonical LABELS order"),
    ("duplicate tags", _row(0, target_tags=["cs", "cs"]), "duplicate target_tags"),
    ("tag outside LABELS", _row(0, target_tags=["other"]), "invalid target_tags"),
    ("proposal flag", _row(0, is_layer2_proposal=False), "is_layer2_proposal inconsistent"),
    ("multi target flag", _row(2, is_multi_target=False), "is_multi_target inconsistent"),
    ("error flag", _row(1, quality_flags=["time_invalid"]), "error-class quality flags"),
    ("general tags", _row(0, general_tags="x"), "invalid general_tags"),
    ("invalid split", _row(0, split="dev"), "invalid split"),
    ("test fold", _row(5, fold=0), "must have fold=null"),
    ("fold range", _row(0, fold=9), "missing a valid fold"),
    ("two splits", _row(1, split="val", fold=1), "multiple splits or folds"),
    ("declared disagrees", _declared_meeting("m1", fold=3), "disagrees with splits.json"),
    ("meeting without rows", _declared_meeting("m9", split="train", fold=2), "split meeting coverage mismatch"),
]

LINK_CASES = [
    ("missing link field", _delete_link_field, "missing mandatory fields"),
    ("duplicate link", _duplicate_link, "duplicate link record"),
    ("missing proposal", _link(0, proposal_id="ghost"), "missing proposal seg_id"),
    ("ineligible proposal", _link(0, proposal_id="m1-r"), "ineligible proposal"),
    ("status", _link(0, status="weird"), "invalid status"),
    ("has_response", _link(0, has_response=False), "has_response inconsistent"),
    ("no_pair with response", _link(0, status="no_pair"), "cannot have a response"),
    ("ok without response", _link(0, responder_chains=[], has_response=False), "must have a response"),
    ("missing response", _link(0, responder_chains=[["ghost"]]), "missing response seg_id"),
    ("empty chain", _link(0, responder_chains=[[]]), "empty responder chain"),
    ("cross meeting chain", _link(0, responder_chains=[["m2-r"]]), "crosses meetings"),
    ("chain order", _link(0, responder_chains=[["m1-q", "m1-r"]]), "out of order"),
    ("chains by first position", _link(0, responder_chains=[["m1-q"], ["m1-r"]]), "not ordered by first response position"),
    ("tokens type", _link(0, adjacency_tokens="1a"), "invalid adjacency_tokens"),
    ("pair group type", _link(0, pair_group=5), "invalid pair_group"),
    ("proposal without link", _drop_link, "proposal rows without link records"),
]


def test_valid_parts_pass(parts: Parts) -> None:
    """The release built by make_release satisfies the validator."""
    validate_release_content(*parts)


@pytest.mark.parametrize(("name", "mutate", "message"), ROW_CASES, ids=[case[0] for case in ROW_CASES])
def test_row_and_split_violations_are_rejected(
    parts: Parts, name: str, mutate: Callable[[Parts], None], message: str
) -> None:
    """Each broken row or split rule raises its own ValueError."""
    mutate(parts)
    with pytest.raises(ValueError, match=message):
        validate_release_content(*parts)


@pytest.mark.parametrize(("name", "mutate", "message"), LINK_CASES, ids=[case[0] for case in LINK_CASES])
def test_link_violations_are_rejected(
    parts: Parts, name: str, mutate: Callable[[Parts], None], message: str
) -> None:
    """Each broken link rule raises its own ValueError."""
    mutate(parts)
    with pytest.raises(ValueError, match=message):
        validate_release_content(*parts)


def test_positions_may_have_gaps_where_rows_were_removed(parts: Parts) -> None:
    """Strictly increasing positions with holes are accepted, since cleaning does not renumber."""
    parts[0][2]["position"] = 5
    validate_release_content(*parts)


def test_a_meeting_may_start_above_position_zero(parts: Parts) -> None:
    """A meeting whose first rows were removed still validates."""
    for index, row in enumerate(parts[0][:3]):
        row["position"] = index + 10
    validate_release_content(*parts)


def test_duplicate_positions_are_still_rejected_next_to_gaps(parts: Parts) -> None:
    """Allowing gaps does not allow two rows at one position."""
    parts[0][2]["position"] = 1
    with pytest.raises(ValueError, match="duplicate position in meeting"):
        validate_release_content(*parts)


def test_rows_with_gaps_must_still_be_sorted_by_position(parts: Parts) -> None:
    """Out-of-order rows are rejected even when positions have gaps."""
    parts[0][0]["position"] = 9
    with pytest.raises(ValueError, match=r"sorted by \(meeting, position\)"):
        validate_release_content(*parts)


def test_loaded_release_exposes_validated_content(built: Path) -> None:
    """load_release returns rows, links, splits and metadata with the right sizes."""
    release = load_release(built)
    assert (len(release.rows), len(release.links)) == (6, 3)
    assert release.metadata["data_version"] == "v1"
    assert release.splits["k"] == 5


def test_loaded_release_is_immutable(built: Path) -> None:
    """The release container is frozen and its rows are a tuple."""
    release = load_release(built)
    assert isinstance(release.rows, tuple) and isinstance(release.links, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        release.rows = ()


def test_loader_accepts_exactly_the_five_documented_files(built: Path) -> None:
    """A valid release holds exactly the five documented file names."""
    assert {path.name for path in built.iterdir()} == RELEASE_FILES


def test_missing_release_directory_is_rejected(tmp_path: Path) -> None:
    """A release path that does not exist raises a clear error."""
    with pytest.raises(ValueError, match="does not exist"):
        load_release(tmp_path / "nowhere")


@pytest.mark.parametrize("name", sorted(RELEASE_FILES))
def test_missing_file_is_rejected(built: Path, name: str) -> None:
    """Removing any one of the five files is rejected."""
    (built / name).unlink()
    with pytest.raises(ValueError, match="exactly five files"):
        load_release(built)


def test_extra_file_is_rejected(built: Path) -> None:
    """An additional file in the release folder is rejected."""
    (built / "notes.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly five files"):
        load_release(built)


@pytest.mark.parametrize("name", ["rows.jsonl", "links.jsonl", "splits.json", "metadata.json"])
def test_tampered_content_fails_its_checksum(built: Path, name: str) -> None:
    """Changing a content file after the build is caught by the checksum."""
    path = built / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=f"checksum mismatch for {name}"):
        load_release(built)


def test_checksums_must_not_list_themselves(built: Path) -> None:
    """checksums.sha256 listing itself is rejected."""
    path = built / "checksums.sha256"
    path.write_text(path.read_text(encoding="ascii") + f"{'0' * 64}  checksums.sha256\n", encoding="ascii")
    with pytest.raises(ValueError, match="unexpected checksum filename"):
        load_release(built)


def test_checksums_must_list_all_four_content_files(built: Path) -> None:
    """A checksum file with a missing entry is rejected."""
    path = built / "checksums.sha256"
    path.write_text("".join(path.read_text(encoding="ascii").splitlines(keepends=True)[:3]), encoding="ascii")
    with pytest.raises(ValueError, match="exactly the four content files"):
        load_release(built)


def test_checksum_entries_must_be_sorted(built: Path) -> None:
    """Checksum lines out of filename order are rejected."""
    path = built / "checksums.sha256"
    lines = path.read_text(encoding="ascii").splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="sorted by filename"):
        load_release(built)


def test_duplicate_and_malformed_checksum_lines_are_rejected(built: Path) -> None:
    """Duplicate entries and bad digests are rejected."""
    path = built / "checksums.sha256"
    original = path.read_text(encoding="ascii")
    path.write_text(original + original.splitlines()[0] + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="duplicate checksum entry"):
        load_release(built)
    path.write_text(original.replace(original[:64], "G" * 64, 1), encoding="ascii")
    with pytest.raises(ValueError, match="invalid checksum line"):
        load_release(built)


def test_blank_jsonl_line_is_rejected(built: Path) -> None:
    """A blank record inside rows.jsonl is rejected even when the checksum matches."""
    path = built / "rows.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rewrite_checksums(built)
    with pytest.raises(ValueError, match="blank JSONL record"):
        load_release(built)


def test_invalid_json_is_rejected_with_the_file_name(built: Path) -> None:
    """Broken JSON in a content file names the file in the error."""
    (built / "splits.json").write_text("{not json", encoding="utf-8")
    rewrite_checksums(built)
    with pytest.raises(ValueError, match="splits.json"):
        load_release(built)


@pytest.mark.parametrize("version", ["1", "v0", "V1", "latest", "v1.0", 1, None])
def test_data_version_must_look_like_vn(built: Path, version: Any) -> None:
    """metadata.data_version must be v1, v2 and so on."""
    metadata = json.loads((built / "metadata.json").read_text(encoding="utf-8"))
    metadata["data_version"] = version
    (built / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rewrite_checksums(built)
    with pytest.raises(ValueError, match="data_version"):
        load_release(built)
