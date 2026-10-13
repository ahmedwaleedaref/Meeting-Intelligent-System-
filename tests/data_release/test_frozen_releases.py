"""Task 3: every release folder found under data/ (checksums, provenance, metadata, full load)."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from smi.data.release import RELEASE_FILES, ReleaseData, load_release
from smi.inputs.formatter import ContextFormatter
from smi.labels import LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = PROJECT_ROOT / "tasks" / "A-data-transformation" / "provenance"
DATA_ROOT = PROJECT_ROOT / "data"
VERSIONS = tuple(sorted(path.name for path in DATA_ROOT.glob("v[0-9]*") if path.is_dir()))
CONTENT_FILES = ("links.jsonl", "metadata.json", "rows.jsonl", "splits.json")


def release_dir(version: str) -> Path:
    return DATA_ROOT / version


def rows_are_available(version: str) -> bool:
    path = release_dir(version) / "rows.jsonl"
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return not stream.read(32).startswith(b"version https://git-lfs")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checksum_entries(version: str) -> dict[str, str]:
    lines = (release_dir(version) / "checksums.sha256").read_text(encoding="ascii").splitlines()
    return {line.split("  ")[1]: line.split("  ")[0] for line in lines}


@pytest.fixture(scope="module", params=VERSIONS)
def version(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module", params=VERSIONS)
def release(request: pytest.FixtureRequest) -> ReleaseData:
    if not rows_are_available(request.param):
        pytest.skip(f"data/{request.param}/rows.jsonl is missing or still a Git LFS pointer (run: git lfs pull)")
    return load_release(release_dir(request.param))


def test_release_folder_holds_exactly_the_five_files(version: str) -> None:
    """Each frozen release folder contains the five documented files and nothing more."""
    assert {path.name for path in release_dir(version).iterdir()} == RELEASE_FILES


def test_checksum_file_lists_only_the_four_content_files_sorted(version: str) -> None:
    """checksums.sha256 names the four content files in order and never itself."""
    assert list(checksum_entries(version)) == sorted(CONTENT_FILES)


def test_small_content_files_match_their_recorded_checksums(version: str) -> None:
    """links, metadata and splits hash to the values in checksums.sha256."""
    entries = checksum_entries(version)
    for name in ("links.jsonl", "metadata.json", "splits.json"):
        assert digest(release_dir(version) / name) == entries[name]


def tracked_provenance(version: str) -> dict:
    path = PROVENANCE / f"release_{version}.json"
    if not path.is_file():
        pytest.skip(f"no committed release_{version}.json for this local build")
    return json.loads(path.read_text(encoding="utf-8"))


def test_release_provenance_fingerprint_is_the_hash_of_the_checksum_file(version: str) -> None:
    """release_vN.json, when committed, records the SHA-256 of checksums.sha256 as the fingerprint."""
    record = tracked_provenance(version)
    assert record["data_version"] == version
    assert record["release_fingerprint"] == digest(release_dir(version) / "checksums.sha256")


def test_release_provenance_output_hashes_match_the_release_files(version: str) -> None:
    """The per-file hashes in the committed provenance equal the checksum entries and the checksum file hash."""
    record = tracked_provenance(version)
    hashes = record["release_files_sha256"]
    assert set(hashes) == RELEASE_FILES
    assert {name: hashes[name] for name in CONTENT_FILES} == checksum_entries(version)
    assert hashes["checksums.sha256"] == record["release_fingerprint"]


def test_provenance_inputs_that_are_committed_match_their_recorded_hashes(version: str) -> None:
    """Committed provenance inputs hash to the values recorded at build time, up to line endings."""
    record = tracked_provenance(version)
    for name, recorded in record["input_sha256"].items():
        path = PROVENANCE / name
        if path.is_file():
            lf = path.read_bytes().replace(b"\r\n", b"\n")
            assert recorded in {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}, name


def test_metadata_describes_the_release_without_build_information(version: str) -> None:
    """metadata.json names its version, licence and defaults and holds no timestamp or commit."""
    text = (release_dir(version) / "metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(text)
    assert metadata["data_version"] == version and metadata["license"] == "CC BY 4.0"
    assert set(metadata["defaults"]) == {f"D{number}" for number in range(1, 17)}
    assert re.search(r"\d{4}-\d{2}-\d{2}", text) is None
    assert re.search(r"\b[0-9a-f]{40}\b", text) is None


def test_metadata_counts_are_internally_consistent(version: str) -> None:
    """Split, label and link counts each add up to the stated totals."""
    counts = json.loads((release_dir(version) / "metadata.json").read_text(encoding="utf-8"))["counts"]
    assert sum(counts["rows_by_split"].values()) == counts["total_rows"]
    assert sum(counts["rows_by_label"].values()) == counts["total_rows"]
    assert sum(counts["links_by_status"].values()) == counts["total_links"]
    assert set(counts["rows_by_label"]) == set(LABELS)


def test_links_file_matches_the_metadata_status_counts(version: str) -> None:
    """links.jsonl has one record per counted link and the same status totals."""
    lines = (release_dir(version) / "links.jsonl").read_text(encoding="utf-8").splitlines()
    counts = json.loads((release_dir(version) / "metadata.json").read_text(encoding="utf-8"))["counts"]
    statuses = [json.loads(line)["status"] for line in lines]
    assert len(statuses) == counts["total_links"]
    assert {status: statuses.count(status) for status in counts["links_by_status"]} == counts["links_by_status"]


def test_full_release_loads_and_validates(release: ReleaseData) -> None:
    """The checksum-verified release passes every row, link and split rule."""
    counts = release.metadata["counts"]
    assert len(release.rows) == counts["total_rows"] and len(release.links) == counts["total_links"]


def test_row_counts_by_split_and_label_match_the_metadata(release: ReleaseData) -> None:
    """Counts recomputed from the rows equal the counts stored in metadata.json."""
    counts = release.metadata["counts"]
    by_split: dict[str, int] = {}
    by_label = {label: 0 for label in LABELS}
    for row in release.rows:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
        by_label[row["label"]] += 1
    assert by_split == counts["rows_by_split"] and by_label == counts["rows_by_label"]


def test_layer2_gold_excludes_unresolvable_records(release: ReleaseData) -> None:
    """Gold holds exactly the ok and no_pair records and keeps malformed ones out."""
    statuses = release.metadata["counts"]["links_by_status"]
    gold = release.layer2_gold()
    assert len(gold) == statuses["ok"] + statuses["no_pair"]
    assert all(item["link"]["status"] in {"ok", "no_pair"} for item in gold)
    assert all((not item["acceptable_responses"]) == (item["link"]["status"] == "no_pair") for item in gold)


def test_every_proposal_row_has_exactly_one_link(release: ReleaseData) -> None:
    """Link proposal ids are exactly the proposal rows, once each."""
    proposals = [row["seg_id"] for row in release.rows if row["is_layer2_proposal"]]
    assert sorted(link["proposal_id"] for link in release.links) == sorted(proposals)


def test_formatter_runs_on_real_rows_and_returns_the_requested_window(release: ReleaseData) -> None:
    """The formatter builds from a frozen release and returns a bounded, ordered window."""
    formatter = ContextFormatter(release.rows)
    target = next(row["seg_id"] for row in release.rows if row["is_layer2_proposal"] and row["position"] > 10)
    selected = formatter.select_context(target, 5, 5)
    assert target in selected and 1 <= len(selected) <= 11
    rendered = formatter.format(target, 5, 5, "window_letters")
    assert rendered.count("[T]") == 1 and len(rendered.splitlines()) == len(selected)


def test_build_manifest_matches_the_release_when_present(version: str) -> None:
    """The pipeline manifest records this release's fingerprint and file hashes."""
    path = DATA_ROOT / "interim" / "build_manifest.json"
    if not path.is_file():
        pytest.skip("no pipeline manifest for this folder")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["data_version"] != version:
        pytest.skip(f"manifest describes {manifest['data_version']}, not {version}")
    assert manifest["release_fingerprint"] == digest(release_dir(version) / "checksums.sha256")
    for name in RELEASE_FILES:
        assert manifest["files"][f"{version}/{name}"]["sha256"] == digest(release_dir(version) / name)


def test_release_rows_carry_no_non_speech_flags_and_no_empty_text(release: ReleaseData) -> None:
    """Cleaning removed those rows and flags, so none may remain in the release."""
    flags = {"is_nonspeech", "is_nonlabeled", "is_empty"}
    assert not any(flags & set(row) for row in release.rows)
    assert all(row["text"] for row in release.rows)


def test_positions_strictly_increase_within_each_meeting(release: ReleaseData) -> None:
    """Positions keep the original order, with gaps allowed where rows were removed."""
    last: dict[str, int] = {}
    for row in release.rows:
        assert row["position"] > last.get(row["meeting"], -1)
        last[row["meeting"]] = row["position"]
