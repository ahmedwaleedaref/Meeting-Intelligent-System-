"""
Split a frozen release into one JSONL file per meeting, grouped by split.

input  : <release>/rows.jsonl , <release>/metadata.json , <release>/splits.json
output : <output>/{train,val,test}/<meeting>.jsonl , <output>/source.json

one line per segment , sorted by position :
{"seg_id": ..., "position": ..., "speaker": ..., "text": ..., "label": ...}

all checks run before anything is written ; the script fails on the first problem.
files are written to a temporary folder next to <output> that is renamed only when complete ,
so a failed run never leaves a partial <output>.

usage :
PYTHONPATH=src python3 scripts/make_meeting_files.py
PYTHONPATH=src python3 scripts/make_meeting_files.py --release data/v5 --output data/v5_meetings
"""

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

from smi.jsonl import iter_jsonl
from smi.labels import LABELS

SPLITS = ("train", "val", "test")
FLAGS = ("is_nonspeech", "is_empty", "is_nonlabeled")
# the flags are not required keys : a release built by task A's cleaning step (e.g. v3) removed the
# flagged rows and then the flag fields themselves (schema/03_release.md) , so they may be absent
ROW_KEYS = ("seg_id", "meeting", "position", "split", "speaker_id", "text", "label")


def read_rows(lines: Iterable[str]) -> list[dict]:
    """read the lines of rows.jsonl and check each row on its own"""
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for line_no, row in iter_jsonl(lines, where="rows.jsonl "):
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no}: expected a JSON object, got {type(row).__name__}")
        missing = [key for key in ROW_KEYS if key not in row]
        if missing:
            raise ValueError(f"line {line_no}: missing keys {missing}")
        seg_id = row["seg_id"]
        for flag in FLAGS:
            # absent = already cleaned out ; present = must be False (unchanged rule for releases that keep them)
            if flag in row and row[flag] is not False:
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
    """compare counts with metadata.json , and meeting names and each meeting's split with splits.json"""
    counts = metadata["counts"]
    if len(rows) != counts["total_rows"]:
        raise ValueError(f"{len(rows)} rows, metadata says {counts['total_rows']}")
    for split in SPLITS:
        n_split = sum(1 for row in rows if row["split"] == split)
        if n_split != counts["rows_by_split"][split]:
            raise ValueError(f"{split}: {n_split} rows, metadata says {counts['rows_by_split'][split]}")
    if set(meetings) != set(splits_file["meetings"]):
        only_rows = sorted(set(meetings) - set(splits_file["meetings"]))
        only_splits = sorted(set(splits_file["meetings"]) - set(meetings))
        raise ValueError(f"meetings differ: only in rows.jsonl {only_rows}, only in splits.json {only_splits}")
    # splits.json maps each meeting to {"split": ..., "fold": ...} (task A schema/03_release.md)
    wrong_split = [
        f"{meeting}: rows say {meeting_rows[0]['split']!r}, splits.json says {splits_file['meetings'][meeting]['split']!r}"
        for meeting, meeting_rows in sorted(meetings.items())
        if meeting_rows[0]["split"] != splits_file["meetings"][meeting]["split"]
    ]
    if wrong_split:
        raise ValueError(f"{len(wrong_split)} meetings in a different split than splits.json: {wrong_split}")


def write_meetings(meetings: dict[str, list[dict]], output: Path) -> None:
    for split in SPLITS:
        (output / split).mkdir()
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
    partial = args.output.with_name(f".{args.output.name}.partial")
    if partial.exists():
        raise SystemExit(f"{partial} exists (left by an interrupted run); delete it first")

    metadata = json.loads((args.release / "metadata.json").read_text(encoding="utf-8"))
    splits_file = json.loads((args.release / "splits.json").read_text(encoding="utf-8"))
    # read once : the hash in source.json is of exactly the bytes that were parsed
    rows_bytes = (args.release / "rows.jsonl").read_bytes()
    rows = read_rows(rows_bytes.decode("utf-8").split("\n"))
    meetings = group_by_meeting(rows)
    check_totals(rows, meetings, metadata, splits_file)

    source = {
        "data_version": metadata["data_version"],
        "rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
        "n_meetings": len(meetings),
        "n_rows": len(rows),
    }
    partial.mkdir(parents=True)
    try:
        write_meetings(meetings, partial)
        (partial / "source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(partial)
        raise
    partial.rename(args.output)

    for split in SPLITS:
        split_meetings = [m for m, r in meetings.items() if r[0]["split"] == split]
        n_rows = sum(len(meetings[m]) for m in split_meetings)
        print(f"{split:5} {len(split_meetings):3} meetings {n_rows:7} segments")
    print(f"wrote {args.output} from {metadata['data_version']}")


if __name__ == "__main__":
    main()
