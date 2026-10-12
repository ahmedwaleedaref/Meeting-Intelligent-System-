"""Restore C1 rows omitted from a partial Task 2 export using frozen v1 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def complete_rows(
    task2_rows_path: str | Path,
    previous_release_rows_path: str | Path,
    c1_rows_path: str | Path,
    links_path: str | Path,
    output_path: str | Path,
    provenance_path: str | Path,
) -> int:
    """Combine new C2 rows with unchanged v1 rows missing from the new export."""
    task2_path = Path(task2_rows_path)
    previous_path = Path(previous_release_rows_path)
    c1_path = Path(c1_rows_path)
    links_file = Path(links_path)
    output = Path(output_path)
    provenance = Path(provenance_path)

    if output.exists():
        raise ValueError(f"output already exists: {output}")

    task2_by_id = {}
    for row in _iter_jsonl(task2_path):
        seg_id = row.get("seg_id")
        if not isinstance(seg_id, str) or not seg_id:
            raise ValueError("Task 2 row has a missing or invalid seg_id")
        if seg_id in task2_by_id:
            raise ValueError(f"duplicate Task 2 seg_id: {seg_id!r}")
        task2_by_id[seg_id] = row

    c1_ids = set()
    for row in _iter_jsonl(c1_path):
        seg_id = row.get("seg_id")
        if not isinstance(seg_id, str) or not seg_id or seg_id in c1_ids:
            raise ValueError("C1 rows contain missing or duplicate seg_id values")
        c1_ids.add(seg_id)
    if not set(task2_by_id).issubset(c1_ids):
        raise ValueError("Task 2 contains IDs that do not exist in C1")

    proposal_ids = set()
    for link in _iter_jsonl(links_file):
        proposal_id = link.get("proposal_id")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("Task 2 link has a missing or invalid proposal_id")
        if proposal_id in proposal_ids:
            raise ValueError(f"duplicate Task 2 proposal_id: {proposal_id!r}")
        proposal_ids.add(proposal_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    used_task2_ids = set()
    backfilled = 0
    restored_fields: dict[str, int] = {}
    previous_count = 0
    previous_ids = set()
    with previous_path.open("r", encoding="utf-8") as previous, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line_number, line in enumerate(previous, start=1):
            try:
                previous_row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {previous_path.name}:{line_number}: {error}"
                ) from error
            previous_count += 1
            seg_id = previous_row.get("seg_id")
            if not isinstance(seg_id, str) or not seg_id or seg_id in previous_ids:
                raise ValueError("previous release has a missing or duplicate seg_id")
            previous_ids.add(seg_id)
            if seg_id not in c1_ids:
                raise ValueError(f"previous release ID is absent from C1: {seg_id!r}")

            if seg_id in task2_by_id:
                task2_row = task2_by_id[seg_id]
                _require_unchanged_common_fields(previous_row, task2_row)
                row = dict(previous_row)
                row.pop("split", None)
                row.pop("fold", None)
                for field in row.keys() - task2_row.keys():
                    restored_fields[field] = restored_fields.get(field, 0) + 1
                row.update(task2_row)
                used_task2_ids.add(seg_id)
            else:
                if previous_row.get("is_layer2_proposal") is True:
                    raise ValueError(f"missing Task 2 row is a proposal: {seg_id!r}")
                if seg_id in proposal_ids:
                    raise ValueError(f"missing Task 2 row has a link: {seg_id!r}")
                row = dict(previous_row)
                row.pop("split", None)
                row.pop("fold", None)
                backfilled += 1

            destination.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    if previous_ids != c1_ids:
        raise ValueError("previous release does not contain exactly the C1 IDs")
    if used_task2_ids != set(task2_by_id):
        raise ValueError("Task 2 includes rows not found in the previous release")

    record: dict[str, Any] = {
        "stage": "completed_c2_rows",
        "task2_rows": {
            "path": str(task2_path).replace("\\", "/"),
            "sha256": _sha256_file(task2_path),
            "row_count": len(task2_by_id),
        },
        "task2_links": {
            "path": str(links_file).replace("\\", "/"),
            "sha256": _sha256_file(links_file),
            "record_count": len(proposal_ids),
        },
        "c1_rows": {
            "path": str(c1_path).replace("\\", "/"),
            "sha256": _sha256_file(c1_path),
            "row_count": len(c1_ids),
        },
        "previous_release_rows": {
            "path": str(previous_path).replace("\\", "/"),
            "sha256": _sha256_file(previous_path),
            "row_count": previous_count,
            "release_fingerprint": _previous_release_fingerprint(previous_path),
        },
        "output": {
            "path": str(output).replace("\\", "/"),
            "sha256": _sha256_file(output),
            "row_count": previous_count,
            "backfilled_rows": backfilled,
        },
        "method": (
            "Use every current Task 2 row where present. For IDs absent from the new Task 2 export, "
            "reuse the row from frozen v1 after confirming it is not a Layer-2 proposal and has no link. "
            "For overlapping IDs, shared fields must exactly match frozen v1; fields omitted by the new "
            "export are retained from frozen v1."
        ),
        "overlap_fields_restored_from_v1": dict(sorted(restored_fields.items())),
    }
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return backfilled


def _require_unchanged_common_fields(
    previous_row: dict[str, Any], task2_row: dict[str, Any]
) -> None:
    for field, value in task2_row.items():
        if field in {"split", "fold"}:
            continue
        if field not in previous_row or previous_row[field] != value:
            raise ValueError(
                f"Task 2 changed field {field!r} for {task2_row.get('seg_id')!r} "
                "compared with frozen v1"
            )


def _previous_release_fingerprint(rows_path: Path) -> str:
    checksums_path = rows_path.parent / "checksums.sha256"
    return _sha256_file(checksums_path)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path.name}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            yield record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete a partial Task 2 row export without losing meeting context."
    )
    parser.add_argument("--task2-rows", required=True)
    parser.add_argument("--previous-release-rows", required=True)
    parser.add_argument("--c1-rows", required=True)
    parser.add_argument("--links", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()
    backfilled = complete_rows(
        args.task2_rows,
        args.previous_release_rows,
        args.c1_rows,
        args.links,
        args.output,
        args.provenance,
    )
    print(f"Completed C2 rows; backfilled {backfilled} rows from frozen v1.")


if __name__ == "__main__":
    main()
