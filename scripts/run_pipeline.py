"""Run every Part A stage (parse, annotate, clean, split, renumber, release) and build a data release.

    python scripts/run_pipeline.py --raw-root PATH/icsi_core_nxt --archive PATH/ICSI_core_NXT.zip

Intermediate files go to <data-root>/interim and the release, built from the cleaned rows,
to <data-root>/<version>. The last transformation before the release renumbers each meeting's
positions 0..n-1 in the existing order; the cleaned files keep the original positions.
Tracked provenance and reports are read for comparison but never written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotate_rows import annotate_corpus, sha256  # noqa: E402
from make_release import make_release  # noqa: E402
from make_splits import make_splits  # noqa: E402
from smi.data.annotate.cleaning import (  # noqa: E402
    DROP_FLAGS,
    build_cleaning_report,
    check_cleaned,
    clean_links,
    clean_rows,
)
from smi.data.annotate.contract import check_contract  # noqa: E402
from smi.data.annotate.jsonl import read_parsed_rows, write_jsonl  # noqa: E402
from smi.data.annotate.report import build_report  # noqa: E402
from smi.data.nxt.parser import has_fatal_errors, parse_corpus  # noqa: E402
from smi.data.nxt.validation import validate_rows, write_validation_report  # noqa: E402
from smi.data.splits import load_meeting_ids  # noqa: E402

PROVENANCE_DIR = PROJECT_ROOT / "tasks" / "A-data-transformation" / "provenance"
METADATA_IN_ARCHIVE = "ICSI/ICSI-metadata.xml"
DEFAULT_VERSION = "v3"
DECISIONS_FILE = "decision_defaults_v3.json"
FINGERPRINT_CHECKS = (
    ("parsed rows", "c1.json", ("sha256",), "parsed/rows.jsonl"),
    ("annotated rows", "c2.json", ("rows", "sha256"), "annotated/rows.jsonl"),
    ("annotated links", "c2.json", ("links", "sha256"), "annotated/links.jsonl"),
    ("cleaned rows", "c2_clean.json", ("rows", "sha256"), "cleaned/rows.jsonl"),
    ("cleaned links", "c2_clean.json", ("links", "sha256"), "cleaned/links.jsonl"),
)


class PipelineError(Exception):
    """A stage cannot continue; the message says why and what to look at."""


@dataclass(frozen=True)
class PipelineConfig:
    raw_root: Path
    archive: Path
    data_root: Path
    version: str
    provenance_dir: Path = PROVENANCE_DIR

    @property
    def interim(self) -> Path:
        return self.data_root / "interim"

    @property
    def release_dir(self) -> Path:
        return self.data_root / self.version


def log(message: str) -> None:
    print(message, flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_lines(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


# preflight ------------------------------------------------------------------------------


def preflight(config: PipelineConfig) -> None:
    """Fail fast, before any long stage, on missing inputs or an occupied release folder."""
    if not re.fullmatch(r"v[1-9][0-9]*", config.version):
        raise PipelineError(f"--version must look like v3, got {config.version!r}")
    for name in ("source_archive.json", DECISIONS_FILE):
        if not (config.provenance_dir / name).is_file():
            raise PipelineError(f"missing provenance file {config.provenance_dir / name}")
    for folder in ("DialogueActs", "Words", "transcripts"):
        if not (config.raw_root / folder).is_dir():
            raise PipelineError(f"{config.raw_root} has no {folder}/ folder (see --raw-root)")
    if not config.archive.is_file():
        raise PipelineError(f"archive not found: {config.archive} (see --archive)")
    if config.release_dir.exists() and any(config.release_dir.iterdir()):
        raise PipelineError(f"{config.release_dir} already exists; delete it or choose another --version")


def verify_archive(config: PipelineConfig) -> str:
    recorded = read_json(config.provenance_dir / "source_archive.json")["sha256"]
    actual = sha256(config.archive)
    if actual != recorded:
        raise PipelineError(
            f"{config.archive.name} hashes to {actual}, but provenance/source_archive.json records {recorded}"
        )
    return actual


# stages ---------------------------------------------------------------------------------


def stage_parse(config: PipelineConfig) -> Path:
    """Task 1: NXT release to C1 parsed rows."""
    corrections_path = config.provenance_dir / "transcript_corrections.json"
    corrections = read_json(corrections_path) if corrections_path.is_file() else {}
    rows, counts, transcript_problems = parse_corpus(config.raw_root, corrections)
    contract_problems = validate_rows(rows)
    parsed_path = config.interim / "parsed" / "rows.jsonl"
    write_jsonl(rows, parsed_path)
    report = config.interim / "reports" / "parse_validation.md"
    write_validation_report(report, rows, counts, transcript_problems, contract_problems)
    if has_fatal_errors(rows, transcript_problems, contract_problems):
        raise PipelineError(f"parsing found fatal source or contract errors; see {report}")
    log(f"  parsed    {len(rows):>8} rows from {counts['dialogue_act_elements']} dialogue acts")
    return parsed_path


def stage_annotate(config: PipelineConfig, parsed_path: Path) -> tuple[list[Any], list[Any]]:
    """Task 2: C1 rows to C2 annotated rows and adjacency-pair links."""
    parsed = read_parsed_rows(parsed_path)
    annotated, links, notes = annotate_corpus(parsed)
    problems = check_contract(parsed, annotated, links)
    if problems:
        raise PipelineError("annotation broke the C2 contract:\n  " + "\n  ".join(problems))
    folder = config.interim / "annotated"
    write_jsonl(annotated, folder / "rows.jsonl")
    write_jsonl(links, folder / "links.jsonl")
    report = config.interim / "reports" / "annotation_report.md"
    report.write_text(build_report(annotated, links, notes), encoding="utf-8", newline="\n")
    log(f"  annotated {len(annotated):>8} rows, {len(links)} link records")
    return annotated, links


def stage_clean(config: PipelineConfig, annotated: list[Any], links: list[Any]) -> None:
    """Task 2: drop non-speech, non-labeled and empty rows into a separate cleaned output."""
    cleaned = clean_rows(annotated)
    cleaned_links = clean_links(links, annotated)
    problems = check_cleaned(annotated, links, cleaned, cleaned_links)
    if problems:
        raise PipelineError("cleaning broke its contract:\n  " + "\n  ".join(problems))
    folder = config.interim / "cleaned"
    write_jsonl(cleaned, folder / "rows.jsonl")
    write_jsonl(cleaned_links, folder / "links.jsonl")
    report = config.interim / "reports" / "cleaning_report.md"
    report.write_text(
        build_cleaning_report(annotated, links, cleaned, cleaned_links), encoding="utf-8", newline="\n"
    )
    log(f"  cleaned   {len(cleaned):>8} rows ({len(annotated) - len(cleaned)} dropped by {', '.join(DROP_FLAGS)})")


def stage_split(config: PipelineConfig) -> Path:
    """Task 3: meeting-level splits and folds from the archive's own meeting list."""
    folder = config.interim / "splits"
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / "ICSI-metadata.xml"
    with zipfile.ZipFile(config.archive) as archive:
        metadata_path.write_bytes(archive.read(METADATA_IN_ARCHIVE))
    split_path = folder / "split.json"
    make_splits(metadata_path, split_path)
    split_path.write_bytes(split_path.read_bytes().replace(b"\r\n", b"\n"))
    meetings = load_meeting_ids(metadata_path)
    committed = config.provenance_dir / "split_v1.json"
    same = committed.is_file() and read_json(committed) == read_json(split_path)
    note = "same assignment as committed split_v1.json" if same else "differs from (or has no) committed split_v1.json"
    log(f"  splits    {len(meetings):>8} meetings, {note}")
    return split_path


def renumber_positions(rows: list[Any]) -> list[Any]:
    """Give each meeting's rows positions 0..n-1 in their current order; every other field is kept."""
    next_position: dict[str, int] = {}
    renumbered = []
    for row in rows:
        position = next_position.get(row["meeting"], 0)
        next_position[row["meeting"]] = position + 1
        renumbered.append({**row, "position": position})
    return renumbered


def stage_renumber(config: PipelineConfig) -> Path:
    """Final transformation: make every meeting's cleaned positions continuous (0..n-1)."""
    rows = renumber_positions(read_parsed_rows(config.interim / "cleaned" / "rows.jsonl"))
    release_rows = config.interim / "release_input" / "rows.jsonl"
    write_jsonl(rows, release_rows)
    log(f"  renumber  {len(rows):>8} rows, positions 0..n-1 in every meeting")
    return release_rows


def stage_release(config: PipelineConfig, split_path: Path, rows_path: Path) -> str:
    """Task 3: validate and write the five-file release from the renumbered rows."""
    fingerprint = make_release(
        rows_path,
        config.interim / "cleaned" / "links.jsonl",
        split_path,
        config.provenance_dir / "source_archive.json",
        config.provenance_dir / DECISIONS_FILE,
        config.release_dir,
        data_version=config.version,
    )
    log(f"  released  {config.release_dir.name:>8} fingerprint {fingerprint}")
    return fingerprint


# reporting ------------------------------------------------------------------------------


def lookup(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        record = record[key]
    return str(record)


def compare_with_committed(config: PipelineConfig) -> None:
    """Report whether each rebuilt file matches the fingerprint committed in provenance/."""
    log("Reproducibility against committed provenance:")
    for name, filename, keys, relative in FINGERPRINT_CHECKS:
        path = config.provenance_dir / filename
        if not path.is_file():
            log(f"  {name:<16} no committed {filename} to compare with")
            continue
        try:
            expected = lookup(read_json(path), keys)
        except (json.JSONDecodeError, KeyError) as error:
            log(f"  {name:<16} cannot read {filename}: {error}")
            continue
        verdict = "matches" if sha256(config.interim / relative) == expected else "DIFFERS from"
        log(f"  {name:<16} {verdict} committed {filename}")


def write_manifest(config: PipelineConfig, archive_sha: str, fingerprint: str) -> Path:
    """Record every output's hash and size, with relative paths only so reruns are identical."""
    files: dict[str, dict[str, Any]] = {}
    for path in sorted([*config.interim.rglob("*.jsonl"), *config.release_dir.iterdir()]):
        relative = path.relative_to(config.data_root).as_posix()
        entry: dict[str, Any] = {"sha256": sha256(path)}
        if path.suffix == ".jsonl":
            entry["lines"] = count_lines(path)
        files[relative] = entry
    annotated_rows = files["interim/annotated/rows.jsonl"]["lines"]
    cleaned_rows = files["interim/cleaned/rows.jsonl"]["lines"]
    manifest = {
        "data_version": config.version,
        "source_archive_sha256": archive_sha,
        "release_fingerprint": fingerprint,
        "rows_removed_by_cleaning": annotated_rows - cleaned_rows,
        "files": files,
    }
    path = config.interim / "build_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def run_pipeline(config: PipelineConfig) -> str:
    preflight(config)
    log(f"Checking {config.archive.name} against provenance/source_archive.json")
    archive_sha = verify_archive(config)
    log("Running Part A stages:")
    parsed_path = stage_parse(config)
    annotated, links = stage_annotate(config, parsed_path)
    stage_clean(config, annotated, links)
    split_path = stage_split(config)
    release_rows = stage_renumber(config)
    fingerprint = stage_release(config, split_path, release_rows)
    compare_with_committed(config)
    manifest = write_manifest(config, archive_sha, fingerprint)
    log(f"Done. Release: {config.release_dir}  Manifest: {manifest}")
    return fingerprint


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw" / "icsi_core_nxt",
                        help="folder holding DialogueActs, Words and transcripts")
    parser.add_argument("--archive", type=Path, default=PROJECT_ROOT / "data" / "raw" / "ICSI_core_NXT.zip",
                        help="ICSI_core_NXT.zip (hash-checked; supplies the meeting list)")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data",
                        help="folder that receives interim/ and the release")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="release folder name, such as v3")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    config = PipelineConfig(args.raw_root, args.archive, args.data_root, args.version)
    try:
        run_pipeline(config)
    except (PipelineError, ValueError, FileNotFoundError) as error:
        print(f"Pipeline stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
