"""
two-stage (hierarchical) classifier :
stage 1 : "other" vs "not_other" , trained on every segment
stage 2 : the 6 target classes , trained only on segments whose gold label is not "other"
the two are combined into one 7-class prediction per segment (combine).
"""

from smi.labels import LABELS, TARGET_LABELS

OTHER = "other"
NOT_OTHER = "not_other"

STAGE1_LABELS = (OTHER, NOT_OTHER)
STAGE2_LABELS = TARGET_LABELS


def stage1_label_map() -> list[int]:
    """7-class index -> stage-1 class index : "other" -> 0 , every target class -> 1."""
    label_map = []
    for label in LABELS:
        if label == OTHER:
            label_map.append(STAGE1_LABELS.index(OTHER))
        else:
            label_map.append(STAGE1_LABELS.index(NOT_OTHER))
    return label_map


def stage2_label_map() -> list[int]:
    """7-class index -> stage-2 class index : each target class -> its index , "other" -> -100 (ignored by the loss)."""
    label_map = []
    for label in LABELS:
        if label == OTHER:
            label_map.append(-100)
        else:
            label_map.append(STAGE2_LABELS.index(label))
    return label_map


def stage1_gold(gold: dict[str, str]) -> dict[str, str]:
    """7-class gold -> stage-1 gold ("other" / "not_other") , for every segment."""
    result = {}
    for seg_id in gold:
        if gold[seg_id] == OTHER:
            result[seg_id] = OTHER
        else:
            result[seg_id] = NOT_OTHER
    return result


def stage2_gold(gold: dict[str, str]) -> dict[str, str]:
    """7-class gold -> stage-2 gold : only the segments whose gold label is a target class."""
    result = {}
    for seg_id in gold:
        if gold[seg_id] != OTHER:
            result[seg_id] = gold[seg_id]
    return result


def combine(stage1_records: dict[str, dict], stage2_records: dict[str, dict]) -> dict[str, dict]:
    """
    one 7-class record per segment , from both stages' predictions on the SAME segments :
        p(other) = p1(other)
        p(c)     = p1(not_other) * p2(c)     for each of the 6 target classes
    the 7 scores sum to 1 ; label = the class with the highest combined score (the harness requires
    label = argmax of scores). so a segment gets a target class only if p1(not_other) * max p2 > p1(other) ,
    which is stricter than "stage 1 says not_other" (p1(not_other) > 0.5).
    raises ValueError if the two stages do not cover the same seg_ids
    """
    if set(stage1_records) != set(stage2_records):
        raise ValueError("stage 1 and stage 2 predictions do not cover the same seg_ids")

    records = {}
    for seg_id in stage1_records:
        p1 = stage1_records[seg_id]["scores"]
        p2 = stage2_records[seg_id]["scores"]

        scores = {}
        for label in LABELS:
            if label == OTHER:
                scores[label] = p1[OTHER]
            else:
                scores[label] = p1[NOT_OTHER] * p2[label]

        best = LABELS[0]
        for label in LABELS:
            if scores[label] > scores[best]:
                best = label
        records[seg_id] = {"label": best, "scores": scores}
    return records
