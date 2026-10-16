from pathlib import Path

from smi.jsonl import iter_jsonl
from smi.labels import LABELS


def load_gold(path: str | Path, allow_test: bool = False) -> dict[str, str]:
    """
    read gold labels from the per-meeting files (scripts/make_meeting_files.py).
    path       : one meeting file (e.g. data/v4_meetings/val/Bed006.jsonl)
                 or a split folder (e.g. data/v4_meetings/val) , then every *.jsonl in it is read
    allow_test : the test split (a test/ folder , or a file inside one) is read only when True ;
                 evaluate() passes its explicit test flag through (PROTOCOL §7)

    returns {seg_id: label}
    blank lines are skipped , like load_predictions
    raises ValueError on : the test split without allow_test=True ,
                           a folder with no *.jsonl files , no rows read at all ,
                           invalid JSON , a key repeated inside one row ,
                           a row that is not an object with a non-empty string seg_id and a label ,
                           a duplicate seg_id , a label not in LABELS
    """
    path = Path(path)
    # resolve() so that "test/.." tricks and symlinks to the test folder are judged by the real folder
    split_folder = (path if path.is_dir() else path.parent).resolve()
    if split_folder.name == "test" and not allow_test:
        raise ValueError(f"{path}: test split is read only with allow_test=True (PROTOCOL §7)")

    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise ValueError(f"{path}: no *.jsonl files in folder")
    else:
        files = [path]

    gold: dict[str, str] = {}
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line_no, row in iter_jsonl(f, where=f"{file.name} "):
                if not isinstance(row, dict) or "seg_id" not in row or "label" not in row:
                    raise ValueError(f"{file.name} line {line_no}: expected an object with seg_id and label")
                seg_id = row["seg_id"]
                label = row["label"]
                if not isinstance(seg_id, str) or not seg_id:
                    raise ValueError(f"{file.name} line {line_no}: seg_id must be a non-empty string, got {seg_id!r}")
                if seg_id in gold:
                    raise ValueError(f"{file.name} line {line_no}: duplicate seg_id {seg_id!r}")
                if label not in LABELS:
                    raise ValueError(f"{file.name} line {line_no}: label {label!r} not in {LABELS}")
                gold[seg_id] = label

    if not gold:
        raise ValueError(f"{path}: no rows read")
    return gold
