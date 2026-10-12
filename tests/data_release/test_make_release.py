"""Task 3: make_release output files, metadata, checksums, provenance and rejections."""

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from release_fixtures import (
    DECISIONS,
    SOURCE_ARCHIVE,
    build_release,
    c2_fixture,
    link,
    make_row,
    read_release_parts,
    sha256_text,
    write_inputs,
)
from scripts.make_release import main as make_release_main
from smi.data.release import RELEASE_FILES, load_release
from smi.data.splits import build_split_provenance


@pytest.fixture
def built(tmp_path: Path) -> tuple[Path, str, dict[str, Path]]:
    return build_release(tmp_path)


def test_release_holds_exactly_the_five_files(built: tuple[Path, str, dict[str, Path]]) -> None:
    """A build writes the four content files and checksums.sha256 and nothing else."""
    assert {path.name for path in built[0].iterdir()} == RELEASE_FILES


def test_checksums_cover_the_four_content_files_and_not_themselves(built: tuple[Path, str, dict[str, Path]]) -> None:
    """checksums.sha256 lists the four content files by name, sorted, and never itself."""
    output = built[0]
    lines = (output / "checksums.sha256").read_text(encoding="ascii").splitlines()
    names = [line.split("  ")[1] for line in lines]
    assert names == ["links.jsonl", "metadata.json", "rows.jsonl", "splits.json"]
    for line in lines:
        digest, name = line.split("  ")
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert digest == sha256_text(output / name)
    assert (output / "checksums.sha256").read_bytes().endswith(b"\n")


def test_fingerprint_is_the_hash_of_the_checksum_file(built: tuple[Path, str, dict[str, Path]]) -> None:
    """The release fingerprint is the SHA-256 of checksums.sha256."""
    output, fingerprint, _ = built
    assert fingerprint == sha256_text(output / "checksums.sha256")


def test_two_builds_from_separate_input_copies_are_byte_identical(tmp_path: Path) -> None:
    """Rebuilding from independently written inputs gives identical files and fingerprint."""
    first, first_fingerprint, _ = build_release(tmp_path / "a")
    second, second_fingerprint, _ = build_release(tmp_path / "b")
    assert first_fingerprint == second_fingerprint
    for name in RELEASE_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_metadata_has_the_documented_keys_and_no_build_information(built: tuple[Path, str, dict[str, Path]]) -> None:
    """metadata.json has the documented keys and no timestamp, commit, path or host."""
    text = (built[0] / "metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(text)
    assert set(metadata) == {"data_version", "schema_version", "source_archive", "counts", "defaults", "license"}
    assert metadata["data_version"] == "v1" and metadata["license"] == "CC BY 4.0"
    assert metadata["source_archive"] == SOURCE_ARCHIVE
    assert metadata["defaults"] == DECISIONS
    assert re.search(r"\d{4}-\d{2}-\d{2}", text) is None
    assert re.search(r"\b[0-9a-f]{40}\b", text) is None
    assert "/" not in text.replace("CC BY 4.0", "") and "\\\\" not in text
    for forbidden in ("timestamp", "commit", "hostname", "built_at", "path"):
        assert forbidden not in text.lower()


def test_metadata_counts_match_the_rows_and_links(built: tuple[Path, str, dict[str, Path]]) -> None:
    """Counts by split, label, link status and quality flag are computed from the data."""
    counts = json.loads((built[0] / "metadata.json").read_text(encoding="utf-8"))["counts"]
    assert counts["total_rows"] == 6 and counts["total_links"] == 3
    assert counts["rows_by_split"] == {"train": 3, "val": 2, "test": 1}
    assert counts["rows_by_label"] == {"cs": 1, "co": 1, "aa": 0, "bk": 0, "ar": 0, "cc": 1, "other": 3}
    assert counts["links_by_status"] == {"ok": 2, "no_pair": 1, "malformed": 0, "ambiguous": 0}
    assert counts["rows_by_quality_flag"] == {}


def test_splits_file_records_the_hash_of_the_split_provenance(built: tuple[Path, str, dict[str, Path]]) -> None:
    """splits.json carries K, grouping, per-meeting assignments and the provenance hash."""
    output, _, paths = built
    splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    assert splits["source_sha256"] == sha256_text(paths["splits"])
    assert (splits["k"], splits["grouping"]) == (5, "meeting")
    assert splits["meetings"] == {
        "m1": {"split": "train", "fold": 0},
        "m2": {"split": "val", "fold": 1},
        "m3": {"split": "test", "fold": None},
    }


def test_rows_append_split_and_fold_and_keep_every_c2_field(built: tuple[Path, str, dict[str, Path]]) -> None:
    """Release rows hold the C2 fields unchanged in order with split and fold appended."""
    output, _, paths = built
    c2_rows = [json.loads(line) for line in paths["rows"].read_text(encoding="utf-8").splitlines()]
    rows, _, _ = read_release_parts(output)
    for source, row in zip(c2_rows, rows):
        assert list(row) == [*source, "split", "fold"]
        assert all(row[key] == value for key, value in source.items())
    assert [(row["split"], row["fold"]) for row in rows] == [("train", 0)] * 3 + [("val", 1)] * 2 + [("test", None)]


def test_links_are_copied_unchanged(built: tuple[Path, str, dict[str, Path]]) -> None:
    """links.jsonl equals the C2 links record for record."""
    output, _, paths = built
    expected = [json.loads(line) for line in paths["links"].read_text(encoding="utf-8").splitlines()]
    assert read_release_parts(output)[1] == expected


def test_serialization_is_utf8_lf_and_compact(built: tuple[Path, str, dict[str, Path]]) -> None:
    """JSONL files are compact UTF-8 with LF endings; JSON files are sorted and indented."""
    output = built[0]
    for name in ("rows.jsonl", "links.jsonl"):
        data = (output / name).read_bytes()
        assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf") and data.endswith(b"\n")
        assert b'", "' not in data
    for name in ("splits.json", "metadata.json"):
        parsed = json.loads((output / name).read_text(encoding="utf-8"))
        assert (output / name).read_text(encoding="utf-8") == json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_built_release_loads_and_exposes_layer2_gold(built: tuple[Path, str, dict[str, Path]]) -> None:
    """The release passes the loader and yields one gold record per resolvable proposal."""
    release = load_release(built[0])
    assert [item["proposal_id"] for item in release.layer2_gold()] == ["m1-p", "m1-q", "m2-p"]


def test_missing_context_flags_are_backfilled_from_tags_and_text(tmp_path: Path) -> None:
    """is_nonspeech comes from z/x general tags and is_empty from empty text when absent."""
    rows, links, provenance = c2_fixture()
    rows[1]["general_tags"] = ["x"]
    rows[4]["text"] = ""
    for row in rows:
        row.pop("is_nonspeech"), row.pop("is_empty")
    output, _, _ = build_release(tmp_path, inputs=write_inputs(tmp_path / "in", rows, links, provenance))
    built_rows, _, _ = read_release_parts(output)
    assert [row["is_nonspeech"] for row in built_rows] == [False, True, False, False, False, False]
    assert [row["is_empty"] for row in built_rows] == [False, False, False, False, True, False]


def test_provided_context_flags_are_not_overwritten(tmp_path: Path) -> None:
    """Flags already present in the C2 rows are kept as they are."""
    rows, links, provenance = c2_fixture()
    rows[0]["is_nonspeech"] = True
    output, _, _ = build_release(tmp_path, inputs=write_inputs(tmp_path / "in", rows, links, provenance))
    assert read_release_parts(output)[0][0]["is_nonspeech"] is True


def test_real_build_writes_provenance_with_hashes_of_inputs_and_outputs(tmp_path: Path) -> None:
    """A real build records the fingerprint plus input and output hashes by file name."""
    provenance_path = tmp_path / "provenance" / "release_v1.json"
    output, fingerprint, paths = build_release(tmp_path, provenance_path=provenance_path)
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert record["data_version"] == "v1" and record["release_fingerprint"] == fingerprint
    assert set(record["release_files_sha256"]) == RELEASE_FILES
    assert all(record["release_files_sha256"][name] == sha256_text(output / name) for name in RELEASE_FILES)
    assert set(record["input_sha256"]) == {path.name for path in paths.values()}
    assert all(record["input_sha256"][path.name] == sha256_text(path) for path in paths.values())


def test_dry_run_writes_no_provenance(tmp_path: Path) -> None:
    """A dry run builds the release but never writes the provenance record."""
    provenance_path = tmp_path / "provenance" / "release_v1.json"
    build_release(tmp_path, "dry", dry_run=True, provenance_path=provenance_path)
    assert not provenance_path.exists()


def test_dry_run_cannot_target_a_release_directory(tmp_path: Path) -> None:
    """A dry run into a folder named like the version is refused."""
    with pytest.raises(ValueError, match="temporary directory"):
        build_release(tmp_path, "v2", dry_run=True, data_version="v2")


def test_later_versions_are_labelled_in_metadata(tmp_path: Path) -> None:
    """The requested data version lands in metadata.json."""
    output, _, _ = build_release(tmp_path, "v2", data_version="v2")
    assert load_release(output).metadata["data_version"] == "v2"


def test_command_line_prints_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI builds into the requested folder and prints the release fingerprint."""
    paths = write_inputs(tmp_path / "in")
    output = tmp_path / "cli-release"
    monkeypatch.setattr(sys, "argv", [
        "make_release.py", "--rows", str(paths["rows"]), "--links", str(paths["links"]),
        "--splits", str(paths["splits"]), "--source", str(paths["source"]),
        "--decisions", str(paths["decisions"]), "--output", str(output),
        "--provenance", str(tmp_path / "release.json"),
    ])
    make_release_main()
    assert capsys.readouterr().out.strip() == f"Release fingerprint: {sha256_text(output / 'checksums.sha256')}"


def test_non_empty_output_directory_is_refused(tmp_path: Path) -> None:
    """A build never writes into a folder that already has content."""
    target = tmp_path / "taken"
    target.mkdir()
    (target / "old.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        build_release(tmp_path, "taken")


@pytest.mark.parametrize("version", ["1", "v0", "V1", "v1.0", "release"])
def test_data_version_must_look_like_vn(tmp_path: Path, version: str) -> None:
    """Version labels other than v1, v2 and so on are refused."""
    with pytest.raises(ValueError, match="data_version must look like"):
        build_release(tmp_path, data_version=version)


def test_missing_decision_default_is_refused(tmp_path: Path) -> None:
    """All of D1 to D16 must be present in the decision defaults."""
    paths = write_inputs(tmp_path / "in")
    decisions = {key: value for key, value in DECISIONS.items() if key != "D7"}
    paths["decisions"].write_text(json.dumps(decisions), encoding="utf-8")
    with pytest.raises(ValueError, match="missing D7"):
        build_release(tmp_path, inputs=paths)


@pytest.mark.parametrize(
    ("archive", "message"),
    [
        ({"release": "1", "sha256": "b" * 64}, "missing archive_name"),
        ({"archive_name": "", "release": "1", "sha256": "b" * 64}, "archive_name must be a non-empty string"),
        ({"archive_name": "a.zip", "release": "1", "sha256": "B" * 64}, "lowercase SHA-256"),
        ({"archive_name": "a.zip", "release": "1", "sha256": "b" * 63}, "lowercase SHA-256"),
    ],
)
def test_invalid_source_archive_metadata_is_refused(tmp_path: Path, archive: dict[str, Any], message: str) -> None:
    """Archive name, release and a lowercase SHA-256 are required."""
    paths = write_inputs(tmp_path / "in")
    paths["source"].write_text(json.dumps(archive), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_release(tmp_path, inputs=paths)


def test_meeting_without_a_split_assignment_is_refused(tmp_path: Path) -> None:
    """Every meeting in the rows must have a split assignment."""
    rows, links, _ = c2_fixture()
    partial = build_split_provenance(["m1", "m2"], {"train": ["m1"], "val": ["m2"], "test": []}, split_source="fixture")
    with pytest.raises(ValueError, match="no split assignment found for meeting 'm3'"):
        build_release(tmp_path, inputs=write_inputs(tmp_path / "in", rows, links, partial))


def test_rows_that_already_carry_a_split_are_refused(tmp_path: Path) -> None:
    """C2 rows must not arrive with split or fold already set."""
    rows, links, provenance = c2_fixture()
    rows[0]["split"] = "train"
    with pytest.raises(ValueError, match="already contains split or fold"):
        build_release(tmp_path, inputs=write_inputs(tmp_path / "in", rows, links, provenance))


def test_validator_failures_stop_the_build_before_anything_is_written(tmp_path: Path) -> None:
    """An error-class flag in the input stops the build and leaves no release folder."""
    rows, links, provenance = c2_fixture()
    rows[0]["quality_flags"] = ["ref_unresolved"]
    with pytest.raises(ValueError, match="error-class quality flags"):
        build_release(tmp_path, inputs=write_inputs(tmp_path / "in", rows, links, provenance))
    assert not (tmp_path / "release").exists()


def test_proposal_without_a_link_stops_the_build(tmp_path: Path) -> None:
    """Every proposal row needs a link record."""
    rows, links, provenance = c2_fixture()
    with pytest.raises(ValueError, match="proposal rows without link records"):
        build_release(tmp_path, inputs=write_inputs(tmp_path / "in", rows, links[:-1], provenance))


def test_link_for_a_non_proposal_stops_the_build(tmp_path: Path) -> None:
    """A link record for a row that is not a proposal is refused."""
    rows, links, provenance = c2_fixture()
    extra = [*links, link("m1-r", [])]
    with pytest.raises(ValueError, match="ineligible proposal"):
        build_release(tmp_path, inputs=write_inputs(tmp_path / "in", rows, extra, provenance))


def test_unknown_label_stops_the_build(tmp_path: Path) -> None:
    """A label outside LABELS is refused at build time."""
    rows, links, provenance = c2_fixture()
    rows[1] = make_row("m1-r", "m1", 1, label="zz")
    with pytest.raises(ValueError, match="outside LABELS"):
        build_release(tmp_path, inputs=write_inputs(tmp_path / "in", rows, links, provenance))
