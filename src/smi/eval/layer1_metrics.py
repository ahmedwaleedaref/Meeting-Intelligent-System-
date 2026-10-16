from smi.labels import LABELS, TARGET_LABELS

LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}


def check_inputs(records: dict[str, dict], gold: dict[str, str]) -> None:
    """
    raises ValueError if predictions and gold do not cover the same seg_ids ,
    or if a predicted or gold label is not in LABELS (it would silently drop out of every class)
    the message gives the number of missing / extra seg_ids and up to 5 examples of each
    """
    missing = set(gold) - set(records)
    extra = set(records) - set(gold)
    if missing or extra:
        raise ValueError(
            "predictions and gold do not have the same seg_ids: "
            f"{len(missing)} missing from predictions {sorted(missing)[:5]}, "
            f"{len(extra)} not in gold {sorted(extra)[:5]}"
        )
    unknown = ({record["label"] for record in records.values()} | set(gold.values())) - set(LABELS)
    if unknown:
        raise ValueError(f"labels not in {LABELS}: {sorted(unknown, key=repr)}")


def prf_from_sets(pred_ids: set[str], gold_ids: set[str]) -> dict:
    """
    P / R / F1 / support for one class.
    pred_ids : ids the model labelled with this class
    gold_ids : ids whose gold label is this class

    returns {"p": float, "r": float, "f1": float, "support": int}
    """
    correct = len(pred_ids & gold_ids)  # predicted as this class and gold is this class
    p = correct / len(pred_ids) if pred_ids else 0.0
    r = correct / len(gold_ids) if gold_ids else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"p": p, "r": r, "f1": f1, "support": len(gold_ids)}


def per_class_metrics(records: dict[str, dict], gold: dict[str, str]) -> dict[str, dict]:
    """
    P / R / F1 / support for all 7 classes.
    records : output of load_predictions, seg_id -> {"label": ..., "scores": ...}
    gold    : seg_id -> gold label

    must raise ValueError if set(records) != set(gold) , or on a label not in LABELS (check_inputs)
    one pass over each dict builds the two id sets of every class , then prf_from_sets per class

    returns {label: {"p", "r", "f1", "support"}} with keys in LABELS order
    """
    check_inputs(records, gold)

    pred_ids: dict[str, set[str]] = {label: set() for label in LABELS}
    gold_ids: dict[str, set[str]] = {label: set() for label in LABELS}
    for seg_id, record in records.items():
        pred_ids[record["label"]].add(seg_id)
    for seg_id, label in gold.items():
        gold_ids[label].add(seg_id)
    return {label: prf_from_sets(pred_ids[label], gold_ids[label]) for label in LABELS}


def macro_f1_6(class_metrics: dict[str, dict]) -> float:
    """mean F1 over the 6 target classes ; raises KeyError if a class is missing"""
    return sum(class_metrics[label]["f1"] for label in TARGET_LABELS) / len(TARGET_LABELS)


def macro_f1_7(class_metrics: dict[str, dict]) -> float:
    """mean F1 over all 7 classes ; raises KeyError if a class is missing"""
    return sum(class_metrics[label]["f1"] for label in LABELS) / len(LABELS)


def confusion_matrix(records: dict[str, dict], gold: dict[str, str]) -> list[list[int]]:
    """
    7x7 raw counts. rows = gold label , columns = predicted label , both in LABELS order.
    counts[i][j] = number of segments with gold LABELS[i] predicted as LABELS[j]
    must raise ValueError if set(records) != set(gold) , or on a label not in LABELS (check_inputs)
    """
    check_inputs(records, gold)

    counts: list[list[int]] = [[0] * len(LABELS) for _ in LABELS]
    for seg_id, gold_label in gold.items():
        counts[LABEL_INDEX[gold_label]][LABEL_INDEX[records[seg_id]["label"]]] += 1
    return counts


def row_normalised(counts: list[list[int]]) -> list[list[float]]:
    """
    each row divided by its sum , so the diagonal is the recall of that class.
    a row with no gold segments (sum 0) stays all 0.0
    """
    normalised: list[list[float]] = []
    for row in counts:
        row_sum = sum(row)
        if row_sum == 0:
            normalised.append([0.0] * len(row))
        else:
            normalised.append([value / row_sum for value in row])
    return normalised
