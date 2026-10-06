"""Build the deterministic five-file C3 dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from smi.data.release import load_release, validate_release_content
from smi.data.splits import attach_split_info
from smi.labels import LABELS


CONTENT_FILES = ("links.jsonl", "metadata.json", "rows.jsonl", "splits.json")
REQUIRED_DECISIONS = tuple(f"D{number}" for number in range(1, 17))
LINK_STATUSES = ("ok", "no_pair", "malformed", "ambiguous")


def make_release(
    rows_path: str | Path,
    links_path: str | Path,
    split_path: str | Path,
    source_path: str | Path,
    decisions_path: str | Path,
    output_path: str | Path,
    *,
    dry_run: bool = False,
    provenance_path: str | Path | None = None,
) -> str:
    """Validate C2 inputs and write the five files for a C3 release.

    A dry run writes to the requested output directory, so point it at a
    temporary directory. The committed provenance files remain untouched.
    """
    output = Path(output_path)
    if dry_run and output.name.lower() == "v1":
        raise ValueError("a dry run must use a temporary directory, not data/v1")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")

    rows = _read_jsonl(Path(rows_path))
    links = _read_jsonl(Path(links_path))
    split_bytes = Path(split_path).read_bytes()
    split_provenance = _read_json_bytes(split_bytes, Path(split_path).name)
    source_archive = _read_json(Path(source_path))
    decisions = _read_json(Path(decisions_path))

    _validate_source_archive(source_archive)
    _validate_decisions(decisions)

    splits = _make_release_splits(split_provenance, split_bytes)
    rows = attach_split_info(rows, split_provenance)
    validate_release_content(rows, links, splits)

    metadata = _make_metadata(rows, links, source_archive, decisions)
    content = {
        "rows.jsonl": _jsonl_text(rows),
        "links.jsonl": _jsonl_text(links),
        "splits.json": _json_text(splits),
        "metadata.json": _json_text(metadata),
    }

    output.mkdir(parents=True, exist_ok=True)
    for filename in CONTENT_FILES:
        (output / filename).write_text(content[filename], encoding="utf-8", newline="\n")

    checksum_lines = []
    for filename in sorted(CONTENT_FILES):
        digest = _sha256_file(output / filename)
        checksum_lines.append(f"{digest}  {filename}")
    checksum_text = "\n".join(checksum_lines) + "\n"
    (output / "checksums.sha256").write_text(checksum_text, encoding="ascii", newline="\n")

    release = load_release(output)
    if len(release.rows) != len(rows):
        raise ValueError("written release row count changed during validation")
    fingerprint = _sha256_file(output / "checksums.sha256")

    if not dry_run and provenance_path is not None:
        _write_release_provenance(
            provenance_path,
            fingerprint,
            output,
            (rows_path, links_path, split_path, source_path, decisions_path),
        )

    return fingerprint


def _write_release_provenance(
    provenance_path: str | Path,
    fingerprint: str,
    release_path: Path,
    input_paths: Sequence[str | Path],
) -> None:
    output_files = {}
    for filename in sorted(CONTENT_FILES + ("checksums.sha256",)):
        output_files[filename] = _sha256_file(release_path / filename)

    input_files = {}
    for path_value in input_paths:
        path = Path(path_value)
        input_files[path.name] = _sha256_file(path)

    record = {
        "data_version": "v1",
        "release_fingerprint": fingerprint,
        "input_sha256": input_files,
        "release_files_sha256": output_files,
    }
    path = Path(provenance_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(record), encoding="utf-8", newline="\n")


def _make_release_splits(
    provenance: Mapping[str, Any], split_bytes: bytes
) -> dict[str, Any]:
    fold_info = provenance.get("folds")
    meeting_records = provenance.get("meetings")
    if not isinstance(fold_info, Mapping) or not isinstance(meeting_records, list):
        raise ValueError("split provenance must contain folds and meetings")

    assignments = {}
    for record in meeting_records:
        if not isinstance(record, Mapping):
            raise ValueError("each split provenance meeting must be an object")
        meeting = record.get("meeting_id")
        if not isinstance(meeting, str) or not meeting:
            raise ValueError("split provenance meeting is missing meeting_id")
        if meeting in assignments:
            raise ValueError(f"duplicate split provenance meeting: {meeting!r}")
        assignments[meeting] = {
            "split": record.get("split"),
            "fold": record.get("fold"),
        }

    return {
        "k": fold_info.get("k"),
        "grouping": fold_info.get("grouping"),
        "meetings": assignments,
        "source_sha256": hashlib.sha256(split_bytes).hexdigest(),
    }


def _make_metadata(
    rows: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
    source_archive: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> dict[str, Any]:
    split_counts = {"train": 0, "val": 0, "test": 0}
    label_counts = {}
    for label in LABELS:
        label_counts[label] = 0
    link_counts = {}
    for status in LINK_STATUSES:
        link_counts[status] = 0
    quality_counts = {}

    for row in rows:
        split_counts[row["split"]] += 1
        label_counts[row["label"]] += 1
        for flag in row["quality_flags"]:
            quality_counts[flag] = quality_counts.get(flag, 0) + 1

    for link in links:
        link_counts[link["status"]] += 1

    return {
        "data_version": "v1",
        "schema_version": "1.0",
        "source_archive": {
            "archive_name": source_archive["archive_name"],
            "release": source_archive["release"],
            "sha256": source_archive["sha256"],
        },
        "counts": {
            "total_rows": len(rows),
            "total_links": len(links),
            "rows_by_split": split_counts,
            "rows_by_label": label_counts,
            "links_by_status": link_counts,
            "rows_by_quality_flag": quality_counts,
        },
        "defaults": dict(decisions),
        "license": "CC BY 4.0",
    }


def _validate_source_archive(source: Mapping[str, Any]) -> None:
    required = ("archive_name", "release", "sha256")
    for field in required:
        if field not in source:
            raise ValueError(f"source archive metadata is missing {field}")
    if not isinstance(source["archive_name"], str) or not source["archive_name"]:
        raise ValueError("source archive archive_name must be a non-empty string")
    if not isinstance(source["release"], str) or not source["release"]:
        raise ValueError("source archive release must be a non-empty string")
    digest = source["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("source archive sha256 must be a lowercase SHA-256 fingerprint")
    for character in digest:
        if character not in "0123456789abcdef":
            raise ValueError("source archive sha256 must be a lowercase SHA-256 fingerprint")


def _validate_decisions(decisions: Mapping[str, Any]) -> None:
    for decision in REQUIRED_DECISIONS:
        if decision not in decisions:
            raise ValueError(f"decision defaults are missing {decision}")


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_bytes(path.read_bytes(), path.name)


def _read_json_bytes(contents: bytes, filename: str) -> dict[str, Any]:
    try:
        value = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {filename}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{filename} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path.name}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number} must be a JSON object")
            records.append(record)
    return records


def _jsonl_text(records: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for record in records:
        lines.append(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate the C3 data/v1 release.")
    parser.add_argument("--rows", required=True, help="C2 annotated rows.jsonl")
    parser.add_argument("--links", required=True, help="C2 links.jsonl")
    parser.add_argument("--splits", required=True, help="provenance/split_v1.json")
    parser.add_argument("--source", required=True, help="provenance/source_archive.json")
    parser.add_argument("--decisions", required=True, help="JSON object containing D1 through D16")
    parser.add_argument("--output", default="data/v1", help="release output directory")
    parser.add_argument(
        "--provenance",
        default="tasks/A-data-transformation/provenance/release_v1.json",
        help="release fingerprint record (skipped for dry runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build a fixture release; use a temporary output directory",
    )
    args = parser.parse_args()

    fingerprint = make_release(
        args.rows,
        args.links,
        args.splits,
        args.source,
        args.decisions,
        args.output,
        dry_run=args.dry_run,
        provenance_path=args.provenance,
    )
    print(f"Release fingerprint: {fingerprint}")


if __name__ == "__main__":
    main()
