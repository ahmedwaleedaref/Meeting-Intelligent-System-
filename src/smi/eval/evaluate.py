import json
from pathlib import Path

from smi.eval.gold import load_gold
from smi.eval.layer1_metrics import confusion_matrix, macro_f1_6, macro_f1_7, per_class_metrics, row_normalised
from smi.eval.prediction_schema import load_predictions
from smi.labels import LABELS


def compute_metrics(records: dict[str, dict], gold: dict[str, str]) -> dict:
    """
    all layer-1 metrics of one set of predictions , in memory (no files).
    records : seg_id -> {"label": ..., "scores": {label: probability} or None} (from predict or load_predictions)
    gold    : seg_id -> gold label (from load_gold)

    used by the training loop every epoch and by evaluate , so the number that selects the epoch
    and the number in metrics.json come from the same code.

    returns {"n_segments", "primary": {"macro_f1_6"}, "per_class", "macro_f1_7",
             "confusion": {"labels", "counts", "row_normalised"}, "pr_auc": None}
    pr_auc is None for now (definition not decided yet).
    raises ValueError if records and gold do not cover the same seg_ids , on a label not in LABELS ,
                      or on a record whose label is not the class with the highest score
    """
    # a label that is not the argmax of its scores means a broken pipeline (the harness is argmax only)
    for seg_id in records:
        record = records[seg_id]
        if record["scores"] is not None:
            highest = max(record["scores"].values())
            if record["scores"][record["label"]] != highest:
                raise ValueError(f"seg_id {seg_id!r}: label {record['label']!r} is not the class with the highest score")

    class_metrics = per_class_metrics(records, gold)
    counts = confusion_matrix(records, gold)

    return {
        "n_segments": len(gold),
        "primary": {"macro_f1_6": macro_f1_6(class_metrics)},
        "per_class": class_metrics,
        "macro_f1_7": macro_f1_7(class_metrics),
        "confusion": {"labels": list(LABELS), "counts": counts, "row_normalised": row_normalised(counts)},
        "pr_auc": None,
    }


def evaluate(pred_path, split_dir, out_path, allow_test: bool = False, extra: dict | None = None) -> dict:
    """
    score one prediction file against the gold of one split , and write metrics.json.
    pred_path  : a B1 prediction file (write_predictions)
    split_dir  : the split folder the predictions are for (e.g. data/v4_meetings/val)
    out_path   : where metrics.json is written
    allow_test : must be True to evaluate the test split (PROTOCOL §7) ; always recorded in the output
    extra      : fields added to the output as they are (e.g. {"selected_epoch": 3} from the training loop) ;
                 they can never replace a computed field

    output = {"split": name of split_dir , "data_version": from source.json next to split_dir ,
              "allow_test": ...} + compute_metrics(...) + extra
    returns the same dict it writes.
    raises FileExistsError if out_path already exists (a run's files are never overwritten , PROTOCOL §4.3) ,
           ValueError if an extra key is already a computed field ,
           and everything load_predictions , load_gold and compute_metrics raise
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path}: already exists, metrics are never overwritten (PROTOCOL §4.3)")

    split_dir = Path(split_dir)
    records = load_predictions(pred_path)
    gold = load_gold(split_dir, allow_test=allow_test)

    # source.json is written next to the split folders by scripts/make_meeting_files.py
    with open(split_dir.parent / "source.json", "r", encoding="utf-8") as source_file:
        source = json.load(source_file)

    result = {
        "split": split_dir.resolve().name,
        "data_version": source["data_version"],
        "allow_test": allow_test,
    }
    metrics = compute_metrics(records, gold)
    for key in metrics:
        result[key] = metrics[key]

    if extra is not None:
        for key in extra:
            if key in result:
                raise ValueError(f"extra field {key!r} would replace a computed field")
            result[key] = extra[key]

    # "x" = create a new file , fail if it already exists
    with open(out_path, "x", encoding="utf-8") as out_file:
        json.dump(result, out_file, indent=2)
    return result
