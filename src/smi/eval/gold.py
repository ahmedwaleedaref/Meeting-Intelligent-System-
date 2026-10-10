import json
from pathlib import Path

from smi.labels import LABELS


def load_gold(path: str | Path) -> dict[str, str]:
    """
    read gold labels from the per-meeting files (scripts/make_meeting_files.py).
    path : one meeting file (e.g. data/v4_meetings/val/Bed003.jsonl)
           or a split folder (e.g. data/v4_meetings/val) , then every *.jsonl in it is read

    returns {seg_id: label}
    raises ValueError on a duplicate seg_id or a label not in LABELS
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        files = [path]

    gold: dict[str, str] = {}
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                seg_id = row["seg_id"]
                label = row["label"]
                if seg_id in gold:
                    raise ValueError(f"{file.name}: duplicate seg_id {seg_id!r}")
                if label not in LABELS:
                    raise ValueError(f"{file.name}: label {label!r} not in {LABELS}")
                gold[seg_id] = label
    return gold
