"""
Split a frozen release into one JSONL file per meeting, grouped by split.

input  : <release>/rows.jsonl , <release>/metadata.json , <release>/splits.json
output : <output>/{train,val,test}/<meeting>.jsonl , <output>/source.json

one line per segment , sorted by position :
{"seg_id": ..., "position": ..., "speaker": ..., "text": ..., "label": ...}

all checks run before anything is written ; the script fails on the first problem.

usage :
PYTHONPATH=src python3 scripts/make_meeting_files.py
PYTHONPATH=src python3 scripts/make_meeting_files.py --release data/v5 --output data/v5_meetings
"""

import argparse
import hashlib
import json
from pathlib import Path

from smi.labels import LABELS

SPLITS = ("train", "val", "test")
FLAGS = ("is_nonspeech", "is_empty", "is_nonlabeled")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(rows_path: Path) -> list[dict]:
    """read rows.jsonl and check each row on its own"""
    rows: list[dict] = []
    seen_ids: set[str] = set()
    with open(rows_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            seg_id = row["seg_id"]
            for flag in FLAGS:
                if row[flag] is not False:
                    raise ValueError(f"{seg_id}: {flag} is {row[flag]!r}, expected False")
            if seg_id in seen_ids:
                raise ValueError(f"line {line_no}: duplicate seg_id {seg_id!r}")
            if row["label"] not in LABELS:
                raise ValueError(f"{seg_id}: label {row['label']!r} not in {LABELS}")
            if row["split"] not in SPLITS:
                raise ValueError(f"{seg_id}: split {row['split']!r} not in {SPLITS}")
            seen_ids.add(seg_id)
            rows.append(row)
    return rows


def group_by_meeting(rows: list[dict]) -> dict[str, list[dict]]:
    """group rows per meeting , sort by position , check split and positions"""
    meetings: dict[str, list[dict]] = {}
    for row in rows:
        if row["meeting"] not in meetings:
            meetings[row["meeting"]] = []
        meetings[row["meeting"]].append(row)

    for meeting, meeting_rows in meetings.items():
        meeting_rows.sort(key=lambda row: row["position"])
        splits = {row["split"] for row in meeting_rows}
        if len(splits) != 1:
            raise ValueError(f"meeting {meeting} is in more than one split: {sorted(splits)}")
        positions = [row["position"] for row in meeting_rows]
        if positions != list(range(len(meeting_rows))):
            raise ValueError(f"meeting {meeting}: positions are not 0..{len(meeting_rows) - 1} without gaps")
    return meetings


def check_totals(rows: list[dict], meetings: dict[str, list[dict]], metadata: dict, splits_file: dict) -> None:
    """compare counts with metadata.json and splits.json"""
    counts = metadata["counts"]
    if len(rows) != counts["total_rows"]:
        raise ValueError(f"{len(rows)} rows, metadata says {counts['total_rows']}")
    for split in SPLITS:
        n_split = sum(1 for row in rows if row["split"] == split)
        if n_split != counts["rows_by_split"][split]:
            raise ValueError(f"{split}: {n_split} rows, metadata says {counts['rows_by_split'][split]}")
    if set(meetings) != set(splits_file["meetings"]):
        raise ValueError(f"{len(meetings)} meetings in rows, {len(splits_file['meetings'])} in splits.json")


def write_meetings(meetings: dict[str, list[dict]], output: Path) -> None:
    for split in SPLITS:
        (output / split).mkdir(parents=True)
    for meeting, meeting_rows in meetings.items():
        split = meeting_rows[0]["split"]
        with open(output / split / f"{meeting}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for row in meeting_rows:
                out = {
                    "seg_id": row["seg_id"],
                    "position": row["position"],
                    "speaker": row["speaker_id"],
                    "text": row["text"],
                    "label": row["label"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=Path("data/v4"))
    parser.add_argument("--output", type=Path, default=Path("data/v4_meetings"))
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; delete it first to regenerate")

    metadata = json.loads((args.release / "metadata.json").read_text(encoding="utf-8"))
    splits_file = json.loads((args.release / "splits.json").read_text(encoding="utf-8"))
    rows = read_rows(args.release / "rows.jsonl")
    meetings = group_by_meeting(rows)
    check_totals(rows, meetings, metadata, splits_file)

    write_meetings(meetings, args.output)
    source = {
        "data_version": metadata["data_version"],
        "rows_sha256": sha256_of(args.release / "rows.jsonl"),
        "n_meetings": len(meetings),
        "n_rows": len(rows),
    }
    (args.output / "source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")

    for split in SPLITS:
        split_meetings = [m for m, r in meetings.items() if r[0]["split"] == split]
        n_rows = sum(len(meetings[m]) for m in split_meetings)
        print(f"{split:5} {len(split_meetings):3} meetings {n_rows:7} segments")
    print(f"wrote {args.output} from {metadata['data_version']}")


if __name__ == "__main__":
    main()
