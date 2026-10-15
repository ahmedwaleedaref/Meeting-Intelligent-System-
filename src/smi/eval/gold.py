import json
from pathlib import Path

from smi.labels import LABELS


def load_gold(path: str | Path) -> dict[str, str]:
    """
    read gold labels from the per-meeting files (scripts/make_meeting_files.py).
    path : one meeting file (e.g. data/v4_meetings/val/Bed006.jsonl)
           or a split folder (e.g. data/v4_meetings/val) , then every *.jsonl in it is read

    returns {seg_id: label}
    blank lines are skipped , like load_predictions
    raises ValueError on : a folder with no *.jsonl files , no rows read at all ,
                           invalid JSON , a duplicate seg_id , a label not in LABELS
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise ValueError(f"{path}: no *.jsonl files in folder")
    else:
        files = [path]

    gold: dict[str, str] = {}
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{file.name} line {line_no}: invalid JSON ({e.msg})") from e
                seg_id = row["seg_id"]
                label = row["label"]
                if seg_id in gold:
                    raise ValueError(f"{file.name} line {line_no}: duplicate seg_id {seg_id!r}")
                if label not in LABELS:
                    raise ValueError(f"{file.name} line {line_no}: label {label!r} not in {LABELS}")
                gold[seg_id] = label

    if not gold:
        raise ValueError(f"{path}: no rows read")
    return gold
