"""Part A: scripts/run_pipeline.py end to end on a hand-built 75-meeting corpus."""

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from nxt_fixtures import act, word, write_corpus
from scripts.run_pipeline import PipelineConfig, PipelineError, main, renumber_positions, run_pipeline
from smi.data.release import load_release

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_PROVENANCE = PROJECT_ROOT / "tasks" / "A-data-transformation" / "provenance"
MEETINGS = [f"Bmr{number:03d}" for number in range(1, 76)]
KEPT_PER_MEETING = 3
REMOVED_PER_MEETING = 3
FLAG_FIELDS = {"is_nonspeech", "is_nonlabeled", "is_empty"}


def build_meeting(raw_root: Path, meeting: str) -> None:
    ids = [f"{meeting}.w.{number}" for number in range(1, 6)]
    words = [word(ids[index], f"w{index}", f"{index * 10}.0", f"{index * 10 + 1}.0") for index in range(5)]

    def href(index: int) -> str:
        return f"{meeting}.A.words.xml#id({ids[index]})"

    specs = [
        ("1", "0.0", "1.0", "s^cs", "1a", 0),
        ("2", "10.0", "11.0", "s^aa", "1b", 1),
        ("3", "20.0", "21.0", "x", None, 2),
        ("4", "30.0", "31.0", "z", None, 3),
        ("5", "40.0", "41.0", "s", None, None),
        ("6", "50.0", "51.0", "s", None, 4),
    ]
    acts = [
        act(
            f"{meeting}.A.dialogueact{number}", start, end, type_, original_type=type_,
            adjacency=adjacency, href=None if index is None else href(index),
        )
        for number, start, end, type_, adjacency, index in specs
    ]
    write_corpus(raw_root, words={"A": words}, acts={"A": acts}, meeting=meeting)


def build_archive(path: Path) -> None:
    observations = "".join(f"<observation name='{meeting}'/>" for meeting in MEETINGS)
    xml = f"<corpus><observations>{observations}</observations></corpus>"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ICSI/ICSI-metadata.xml", xml)


def make_provenance(folder: Path, archive: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    record = {"archive_name": archive.name, "release": "fixture-1", "sha256": digest}
    (folder / "source_archive.json").write_text(json.dumps(record), encoding="utf-8")
    shutil.copy(REAL_PROVENANCE / "decision_defaults_v3.json", folder / "decision_defaults_v3.json")
    return folder


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("pipeline-inputs")
    raw_root = root / "raw"
    for meeting in MEETINGS:
        build_meeting(raw_root, meeting)
    archive = root / "ICSI_core_NXT.zip"
    build_archive(archive)
    return {"raw": raw_root, "archive": archive, "provenance": make_provenance(root / "provenance", archive)}


def config_for(inputs: dict[str, Path], data_root: Path, version: str = "v3") -> PipelineConfig:
    return PipelineConfig(inputs["raw"], inputs["archive"], data_root, version, inputs["provenance"])


@pytest.fixture(scope="module")
def built(inputs: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    data_root = tmp_path_factory.mktemp("pipeline-output")
    return data_root, run_pipeline(config_for(inputs, data_root))


def test_pipeline_writes_a_release_that_loads_and_validates(built: tuple[Path, str]) -> None:
    """The release folder holds the five files and passes the loader's validation."""
    data_root, fingerprint = built
    release = load_release(data_root / "v3")
    assert len(release.rows) == len(MEETINGS) * KEPT_PER_MEETING
    assert len(release.links) == len(MEETINGS)
    assert fingerprint == hashlib.sha256((data_root / "v3" / "checksums.sha256").read_bytes()).hexdigest()


def test_release_is_built_from_cleaned_rows_without_flags_or_empty_text(built: tuple[Path, str]) -> None:
    """Cleaning removed the x, z and empty rows and their flags, so none reach the release."""
    release = load_release(built[0] / "v3")
    assert not any(FLAG_FIELDS & set(row) for row in release.rows)
    assert all(row["text"] for row in release.rows)
    assert {row["part_type"] for row in release.rows} == {"s^cs", "s^aa", "s"}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_release_positions_are_continuous_from_zero(built: tuple[Path, str]) -> None:
    """The released rows of every meeting have positions 0, 1, 2 with no gaps."""
    release = load_release(built[0] / "v3")
    for meeting in MEETINGS:
        assert [row["position"] for row in release.rows if row["meeting"] == meeting] == [0, 1, 2]


def test_cleaned_files_keep_the_original_positions_with_gaps(built: tuple[Path, str]) -> None:
    """Cleaning itself does not renumber: kept positions are still 0, 1 and 5."""
    cleaned = read_rows(built[0] / "interim" / "cleaned" / "rows.jsonl")
    for meeting in MEETINGS:
        assert [row["position"] for row in cleaned if row["meeting"] == meeting] == [0, 1, 5]


def test_renumbering_keeps_order_ids_and_every_other_field(built: tuple[Path, str]) -> None:
    """Only position differs between the cleaned rows and the released rows."""
    cleaned = read_rows(built[0] / "interim" / "cleaned" / "rows.jsonl")
    released = read_rows(built[0] / "v3" / "rows.jsonl")
    assert [row["seg_id"] for row in released] == [row["seg_id"] for row in cleaned]
    for before, after in zip(cleaned, released):
        assert {key: value for key, value in after.items() if key not in {"position", "split", "fold"}} == {
            key: value for key, value in before.items() if key != "position"
        }


def test_release_links_are_the_cleaned_links_unchanged(built: tuple[Path, str]) -> None:
    """Links refer to seg_id, so renumbering positions leaves them byte-identical."""
    assert (built[0] / "v3" / "links.jsonl").read_bytes() == (built[0] / "interim" / "cleaned" / "links.jsonl").read_bytes()


def test_renumber_positions_restarts_per_meeting_and_keeps_key_order() -> None:
    """Each meeting counts from zero in its current order and only position changes."""
    rows = [
        {"meeting": "a", "position": 3, "x": 1},
        {"meeting": "a", "position": 9, "x": 2},
        {"meeting": "b", "position": 5, "x": 3},
    ]
    result = renumber_positions(rows)
    assert [(row["meeting"], row["position"]) for row in result] == [("a", 0), ("a", 1), ("b", 0)]
    assert [list(row) for row in result] == [["meeting", "position", "x"]] * 3
    assert rows[0]["position"] == 3


def test_links_survive_cleaning_and_feed_the_gold(built: tuple[Path, str]) -> None:
    """Each meeting's proposal keeps its response, so the gold has one ok record per meeting."""
    release = load_release(built[0] / "v3")
    gold = release.layer2_gold()
    assert len(gold) == len(MEETINGS)
    assert all(item["link"]["status"] == "ok" and len(item["acceptable_responses"]) == 1 for item in gold)


def test_every_stage_writes_its_output_under_interim(built: tuple[Path, str]) -> None:
    """Parsed, annotated and cleaned files, three reports and the split inputs are all produced."""
    interim = built[0] / "interim"
    for relative in (
        "parsed/rows.jsonl", "annotated/rows.jsonl", "annotated/links.jsonl", "cleaned/rows.jsonl",
        "cleaned/links.jsonl", "release_input/rows.jsonl", "reports/parse_validation.md", "reports/annotation_report.md",
        "reports/cleaning_report.md", "splits/split.json", "splits/ICSI-metadata.xml", "build_manifest.json",
    ):
        assert (interim / relative).is_file(), relative


def test_row_counts_shrink_only_at_the_cleaning_stage(built: tuple[Path, str]) -> None:
    """Parse and annotate keep all rows; cleaning removes exactly the x, z and empty rows."""
    interim = built[0] / "interim"
    counts = {name: sum(1 for _ in (interim / name / "rows.jsonl").open("rb")) for name in ("parsed", "annotated", "cleaned")}
    total = len(MEETINGS) * (KEPT_PER_MEETING + REMOVED_PER_MEETING)
    assert counts == {"parsed": total, "annotated": total, "cleaned": len(MEETINGS) * KEPT_PER_MEETING}


def test_manifest_records_hashes_and_the_cleaning_accounting(built: tuple[Path, str]) -> None:
    """The manifest holds the release fingerprint, every file hash and the removed-row count."""
    data_root, fingerprint = built
    text = (data_root / "interim" / "build_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert manifest["release_fingerprint"] == fingerprint and manifest["data_version"] == "v3"
    assert manifest["rows_removed_by_cleaning"] == len(MEETINGS) * REMOVED_PER_MEETING
    assert manifest["files"]["v3/rows.jsonl"]["lines"] == len(MEETINGS) * KEPT_PER_MEETING
    assert str(data_root) not in text


def test_release_metadata_uses_the_cleaned_decision_defaults(built: tuple[Path, str]) -> None:
    """The release states that cleaning removes the rows, not the old keep-every-row default."""
    metadata = json.loads((built[0] / "v3" / "metadata.json").read_text(encoding="utf-8"))
    assert "removes rows" in metadata["defaults"]["D5"]
    assert "Keep every row" not in metadata["defaults"]["D5"]


def test_split_uses_the_archive_meeting_list(built: tuple[Path, str]) -> None:
    """Splits come from the archive's own meeting list: 51 train, 12 val, 12 test meetings."""
    splits = json.loads((built[0] / "v3" / "splits.json").read_text(encoding="utf-8"))["meetings"]
    sizes = [sum(1 for item in splits.values() if item["split"] == name) for name in ("train", "val", "test")]
    assert sizes == [51, 12, 12]


def test_two_runs_give_byte_identical_releases_and_manifests(
    inputs: dict[str, Path], built: tuple[Path, str], tmp_path: Path
) -> None:
    """Re-running on the same inputs reproduces every release file and the manifest exactly."""
    fingerprint = run_pipeline(config_for(inputs, tmp_path))
    assert fingerprint == built[1]
    for name in ("rows.jsonl", "links.jsonl", "splits.json", "metadata.json", "checksums.sha256"):
        assert (tmp_path / "v3" / name).read_bytes() == (built[0] / "v3" / name).read_bytes()
    assert (tmp_path / "interim" / "build_manifest.json").read_bytes() == (built[0] / "interim" / "build_manifest.json").read_bytes()


def test_provenance_inputs_are_never_written(inputs: dict[str, Path], tmp_path: Path) -> None:
    """The provenance folder is read for hashes and defaults but left exactly as it was."""
    before = {path.name: path.read_bytes() for path in inputs["provenance"].iterdir()}
    run_pipeline(config_for(inputs, tmp_path))
    assert {path.name: path.read_bytes() for path in inputs["provenance"].iterdir()} == before


@pytest.mark.parametrize("version", ["3", "v0", "V3", "release"])
def test_bad_version_is_refused_before_any_work(inputs: dict[str, Path], tmp_path: Path, version: str) -> None:
    """A version label that is not v1, v2 and so on stops the pipeline before it writes anything."""
    with pytest.raises(PipelineError, match="--version"):
        run_pipeline(config_for(inputs, tmp_path, version))
    assert not (tmp_path / "interim").exists()


def test_missing_raw_folder_is_refused_before_any_work(inputs: dict[str, Path], tmp_path: Path) -> None:
    """A raw root without the expected folders is refused with a clear message."""
    config = PipelineConfig(tmp_path / "nowhere", inputs["archive"], tmp_path / "out", "v3", inputs["provenance"])
    with pytest.raises(PipelineError, match="DialogueActs"):
        run_pipeline(config)
    assert not (tmp_path / "out").exists()


def test_archive_with_the_wrong_hash_is_refused_before_any_work(inputs: dict[str, Path], tmp_path: Path) -> None:
    """An archive that does not match provenance/source_archive.json stops the run."""
    other = tmp_path / "other.zip"
    build_archive(other)
    other.write_bytes(other.read_bytes() + b"\x00")
    config = PipelineConfig(inputs["raw"], other, tmp_path / "out", "v3", inputs["provenance"])
    with pytest.raises(PipelineError, match="hashes to"):
        run_pipeline(config)
    assert not (tmp_path / "out").exists()


def test_existing_release_folder_is_never_overwritten(inputs: dict[str, Path], tmp_path: Path) -> None:
    """A release folder that already has content is refused before any stage runs."""
    (tmp_path / "v3").mkdir()
    (tmp_path / "v3" / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(PipelineError, match="already exists"):
        run_pipeline(config_for(inputs, tmp_path))
    assert (tmp_path / "v3" / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert not (tmp_path / "interim").exists()


def test_command_line_reports_a_stopped_run_with_exit_code_1(
    monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() prints the reason on stderr and returns 1 instead of raising."""
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline.py", "--raw-root", str(tmp_path / "nowhere"), "--archive", str(inputs["archive"]),
        "--data-root", str(tmp_path / "out"),
    ])
    assert main() == 1
    assert "Pipeline stopped" in capsys.readouterr().err
