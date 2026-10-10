"""Tests for src/smi/eval/layer1_metrics.py.

Covers every test listed in tasks/B-eval-harness/task.md:
  1. perfect predictions        -> every F1 = 1.0
  2. all `other`                -> macro-F1 (6) = 0.0
  3. hand-built toy file        -> exact expected values 
  4. oracle                     -> segments mislabelled by hand, F1 computed by hand
  5. sklearn cross-check        -> every number equals sklearn.metrics on random predictions
  6. error cases                -> mismatched seg_ids raise ValueError
(The B1 validation errors are tested in test_prediction_schema.py.)

Run from the repo root:   python -m pytest tests/eval -v
"""

import json  
import random 

import pytest 
import sklearn.metrics as sk  # required (requirements.txt): a missing sklearn must fail, not skip

from smi.eval.layer1_metrics import (  
    confusion_matrix, 
    macro_f1_6, 
    macro_f1_7, 
    per_class_metrics,  
    prf_from_sets,  
    row_normalised,  
)
from smi.eval.prediction_schema import load_predictions 
from smi.labels import LABELS, TARGET_LABELS 


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def as_records(predicted: dict[str, str]) -> dict[str, dict]:
    """Turn {seg_id: label} into the shape load_predictions returns: {seg_id: {"label", "scores"}}."""
    return {seg_id: {"label": label, "scores": None} for seg_id, label in predicted.items()}


def assert_matrix_close(actual: list[list[float]], expected: list[list[float]]) -> None:
    """Check two matrices are equal row by row, allowing tiny float rounding differences."""
    assert len(actual) == len(expected)  # same number of rows
    for actual_row, expected_row in zip(actual, expected):  # walk both matrices row by row
        assert actual_row == pytest.approx(expected_row)  # approx: 1/3 written two ways still counts as equal


# ---------------------------------------------------------------------------
# Hand-built toy set: 20 segments from 3 meetings, all 7 classes present in gold.
# seg_ids are short on purpose: the harness treats seg_id as an opaque join key.
# ---------------------------------------------------------------------------

TOY_GOLD = {  # seg_id -> the correct (gold) label
    # meeting Bdb001
    "Bdb001_01": "cs", "Bdb001_02": "aa", "Bdb001_03": "other", "Bdb001_04": "co",
    "Bdb001_05": "cc", "Bdb001_06": "bk", "Bdb001_07": "other",
    # meeting Bed002
    "Bed002_08": "cs", "Bed002_09": "ar", "Bed002_10": "other", "Bed002_11": "aa",
    "Bed002_12": "bk", "Bed002_13": "other", "Bed002_14": "cs",
    # meeting Bmr001
    "Bmr001_15": "co", "Bmr001_16": "aa", "Bmr001_17": "other", "Bmr001_18": "bk",
    "Bmr001_19": "other", "Bmr001_20": "ar",
}

# Oracle: the gold above, with 7 segments mislabelled BY HAND (marked "wrong").
TOY_PRED = dict(TOY_GOLD) | {  # start from a copy of gold, then overwrite the mistakes
    "Bdb001_05": "cs",     # wrong: gold cc
    "Bed002_08": "co",     # wrong: gold cs
    "Bed002_10": "cs",     # wrong: gold other
    "Bed002_11": "bk",     # wrong: gold aa
    "Bed002_13": "aa",     # wrong: gold other
    "Bed002_14": "other",  # wrong: gold cs
    "Bmr001_20": "other",  # wrong: gold ar
}

# Hand calculation (segment numbers only):
# class | predicted as this class | gold is this class    | correct | P   | R   | F1 = 2PR/(P+R)
# cs    | 01, 05, 10              | 01, 08, 14            | 1       | 1/3 | 1/3 | 1/3
# co    | 04, 08, 15              | 04, 15                | 2       | 2/3 | 1   | 4/5
# aa    | 02, 13, 16              | 02, 11, 16            | 2       | 2/3 | 2/3 | 2/3
# bk    | 06, 11, 12, 18          | 06, 12, 18            | 3       | 3/4 | 1   | 6/7
# ar    | 09                      | 09, 20                | 1       | 1   | 1/2 | 2/3
# cc    | (none)                  | 05                    | 0       | 0   | 0   | 0
# other | 03, 07, 14, 17, 19, 20  | 03, 07, 10, 13, 17, 19| 4       | 2/3 | 2/3 | 2/3

TOY_EXPECTED = {
    "cs": {"p": 1 / 3, "r": 1 / 3, "f1": 1 / 3, "support": 3},
    "co": {"p": 2 / 3, "r": 1.0, "f1": 4 / 5, "support": 2},
    "aa": {"p": 2 / 3, "r": 2 / 3, "f1": 2 / 3, "support": 3},
    "bk": {"p": 3 / 4, "r": 1.0, "f1": 6 / 7, "support": 3},
    "ar": {"p": 1.0, "r": 1 / 2, "f1": 2 / 3, "support": 2},
    "cc": {"p": 0.0, "r": 0.0, "f1": 0.0, "support": 1},  # never predicted -> P = 0 (zero_division=0)
    "other": {"p": 2 / 3, "r": 2 / 3, "f1": 2 / 3, "support": 6},
}
# macro-F1 (6) = (1/3 + 4/5 + 2/3 + 6/7 + 2/3 + 0) / 6 = (349/105) / 6 = 349/630 ≈ 0.5540
TOY_MACRO_F1_6 = 349 / 630
# macro-F1 (7) = (349/105 + 2/3) / 7 = (419/105) / 7 = 419/735 ≈ 0.5701
TOY_MACRO_F1_7 = 419 / 735

# Confusion matrix by hand: rows = gold, columns = predicted, order cs, co, aa, bk, ar, cc, other.
TOY_CONFUSION = [
    # cs co aa bk ar cc other   <- predicted
    [1, 1, 0, 0, 0, 0, 1],  # gold cs:    01->cs, 08->co, 14->other
    [0, 2, 0, 0, 0, 0, 0],  # gold co:    04, 15 both correct
    [0, 0, 2, 1, 0, 0, 0],  # gold aa:    02, 16 correct, 11->bk
    [0, 0, 0, 3, 0, 0, 0],  # gold bk:    06, 12, 18 all correct
    [0, 0, 0, 0, 1, 0, 1],  # gold ar:    09 correct, 20->other
    [1, 0, 0, 0, 0, 0, 0],  # gold cc:    05->cs
    [1, 0, 1, 0, 0, 0, 4],  # gold other: 10->cs, 13->aa, the other 4 correct
]
# Each row divided by its total (3, 2, 3, 3, 2, 1, 6). The diagonal is each class's recall.
TOY_ROW_NORMALISED = [
    [1 / 3, 1 / 3, 0, 0, 0, 0, 1 / 3],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 2 / 3, 1 / 3, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1 / 2, 0, 1 / 2],
    [1, 0, 0, 0, 0, 0, 0],
    [1 / 6, 0, 1 / 6, 0, 0, 0, 4 / 6],
]


# ---------------------------------------------------------------------------
# prf_from_sets: one class at a time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pred_ids, gold_ids, expected", [
    # half right both ways
    pytest.param({"s1", "s4"}, {"s1", "s2"}, {"p": 0.5, "r": 0.5, "f1": 0.5, "support": 2}, id="half-right"),
    # P and R differ, so swapping them in the code would be caught
    pytest.param({"s1"}, {"s1", "s2", "s3", "s4"}, {"p": 1.0, "r": 0.25, "f1": 0.4, "support": 4}, id="high-p-low-r"),
    pytest.param({"s1", "s2", "s3", "s4"}, {"s1"}, {"p": 0.25, "r": 1.0, "f1": 0.4, "support": 1}, id="low-p-high-r"),
    # no overlap at all
    pytest.param({"s1"}, {"s2"}, {"p": 0.0, "r": 0.0, "f1": 0.0, "support": 1}, id="all-wrong"),
    # division-by-zero cases: must give 0.0, never crash or NaN (zero_division=0)
    pytest.param(set(), {"s1"}, {"p": 0.0, "r": 0.0, "f1": 0.0, "support": 1}, id="never-predicted"),
    pytest.param({"s1"}, set(), {"p": 0.0, "r": 0.0, "f1": 0.0, "support": 0}, id="no-gold"),
    pytest.param(set(), set(), {"p": 0.0, "r": 0.0, "f1": 0.0, "support": 0}, id="nothing-at-all"),
])
def test_prf_from_sets(pred_ids, gold_ids, expected):
    assert prf_from_sets(pred_ids, gold_ids) == pytest.approx(expected)  # approx also checks the keys are the same


# ---------------------------------------------------------------------------
# per_class_metrics: shape and errors
# ---------------------------------------------------------------------------

def test_per_class_keys_follow_labels_order():
    metrics = per_class_metrics(as_records(TOY_PRED), TOY_GOLD)

    assert list(metrics) == list(LABELS)  # same 7 classes, in the fixed LABELS order


@pytest.mark.parametrize("metric_function", [per_class_metrics, confusion_matrix])  # both must refuse a mismatch
@pytest.mark.parametrize("change", ["missing-segment", "extra-segment"])
def test_mismatched_seg_ids_raise(metric_function, change):
    predicted = dict(TOY_PRED)  # copy, so the shared toy data is never modified
    if change == "missing-segment":
        del predicted["Bmr001_20"]  # the model forgot one segment
    else:
        predicted["Bmr001_99"] = "cs"  # the model predicted a segment that is not in gold

    with pytest.raises(ValueError, match="same seg_ids"):
        metric_function(as_records(predicted), TOY_GOLD)


# ---------------------------------------------------------------------------
# test 1: perfect predictions -> every F1 = 1.0
# Decision: TOY_GOLD contains all 7 classes. A class with no gold and no predictions
# correctly gets F1 = 0, which would make "perfect" predictions look imperfect.
# ---------------------------------------------------------------------------

def test_perfect_predictions():
    metrics = per_class_metrics(as_records(TOY_GOLD), TOY_GOLD)  # predict exactly the gold labels

    for label in LABELS:
        assert metrics[label]["p"] == 1.0
        assert metrics[label]["r"] == 1.0
        assert metrics[label]["f1"] == 1.0
    assert macro_f1_6(metrics) == pytest.approx(1.0)
    assert macro_f1_7(metrics) == pytest.approx(1.0)


def test_perfect_predictions_confusion_is_diagonal():
    counts = confusion_matrix(as_records(TOY_GOLD), TOY_GOLD)

    for i, row in enumerate(counts):
        for j, value in enumerate(row):
            if i != j:
                assert value == 0  # nothing off the diagonal = no mistakes
    assert sum(counts[i][i] for i in range(len(LABELS))) == len(TOY_GOLD)  # every segment sits on the diagonal


# ---------------------------------------------------------------------------
# test 2: everything predicted as `other` -> macro-F1 (6) = 0.0
# ---------------------------------------------------------------------------

def test_all_other_gives_zero_primary_metric():
    all_other = {seg_id: "other" for seg_id in TOY_GOLD}  # a model that always says "other"
    metrics = per_class_metrics(as_records(all_other), TOY_GOLD)

    assert macro_f1_6(metrics) == 0.0  # it never finds a single target class
    for label in TARGET_LABELS:
        assert metrics[label]["f1"] == 0.0
    # `other` itself: P = 6/20 (6 of the 20 really are other), R = 6/6 = 1, F1 = 2*0.3*1/1.3 = 6/13
    assert metrics["other"] == pytest.approx({"p": 6 / 20, "r": 1.0, "f1": 6 / 13, "support": 6})
    assert macro_f1_7(metrics) == pytest.approx((6 / 13) / 7)  # only `other` adds to the 7-class mean


# ---------------------------------------------------------------------------
# tests 3 + 4: hand-built toy FILE with hand-computed (oracle) answers.
# Decision: the toy file is generated inside the test from TOY_PRED (not stored in the repo),
# so the data and the expected numbers sit next to each other and cannot drift apart.
# ---------------------------------------------------------------------------

def test_toy_file_end_to_end(tmp_path):
    path = tmp_path / "predictions.jsonl"  # a temporary file, deleted by pytest afterwards
    lines = [json.dumps({"seg_id": seg_id, "label": label, "scores": None}) for seg_id, label in TOY_PRED.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")  # a real prediction file, no gold inside

    records = load_predictions(path)  # B1: read and validate the file
    metrics = per_class_metrics(records, TOY_GOLD)  # B2: score it against gold

    for label in LABELS:
        assert metrics[label] == pytest.approx(TOY_EXPECTED[label]), label  # on failure, the class name is shown
    assert macro_f1_6(metrics) == pytest.approx(TOY_MACRO_F1_6)
    assert macro_f1_7(metrics) == pytest.approx(TOY_MACRO_F1_7)


def test_toy_confusion_matrix():
    counts = confusion_matrix(as_records(TOY_PRED), TOY_GOLD)

    assert counts == TOY_CONFUSION  # whole numbers, so exact equality


def test_toy_row_normalised():
    normalised = row_normalised(TOY_CONFUSION)

    assert_matrix_close(normalised, TOY_ROW_NORMALISED)


def test_row_normalised_diagonal_equals_recall():
    metrics = per_class_metrics(as_records(TOY_PRED), TOY_GOLD)
    normalised = row_normalised(confusion_matrix(as_records(TOY_PRED), TOY_GOLD))

    for i, label in enumerate(LABELS):
        assert normalised[i][i] == pytest.approx(metrics[label]["r"]), label  # the two functions must agree


def test_row_with_no_gold_stays_zero():
    assert row_normalised([[0, 0, 0], [1, 1, 2]]) == [[0.0, 0.0, 0.0], [0.25, 0.25, 0.5]]  # no division by zero


def test_results_do_not_depend_on_line_order():
    reversed_pred = dict(reversed(list(TOY_PRED.items())))  # same predictions, opposite order
    reversed_gold = dict(reversed(list(TOY_GOLD.items())))

    assert per_class_metrics(as_records(reversed_pred), reversed_gold) == per_class_metrics(as_records(TOY_PRED), TOY_GOLD)
    assert confusion_matrix(as_records(reversed_pred), reversed_gold) == TOY_CONFUSION  # matching is by seg_id only


# ---------------------------------------------------------------------------
# macro_f1_6 / macro_f1_7 on hand-written inputs
# ---------------------------------------------------------------------------

def test_macro_f1_6_ignores_other():
    only_other_right = {label: {"f1": 0.0} for label in TARGET_LABELS} | {"other": {"f1": 1.0}}
    only_targets_right = {label: {"f1": 1.0} for label in TARGET_LABELS} | {"other": {"f1": 0.0}}

    assert macro_f1_6(only_other_right) == 0.0  # a perfect `other` must not raise the primary metric
    assert macro_f1_6(only_targets_right) == pytest.approx(1.0)
    assert macro_f1_7(only_targets_right) == pytest.approx(6 / 7)  # the 7-class mean does include `other`


def test_target_labels_are_labels_without_other():
    # macro_f1_6 skips the key "other" but divides by len(TARGET_LABELS);
    # both are only correct together if TARGET_LABELS is exactly LABELS minus "other".
    assert TARGET_LABELS == tuple(label for label in LABELS if label != "other")


# ---------------------------------------------------------------------------
# test 5: every number cross-checked against sklearn.metrics on random predictions.
# Needs scikit-learn (requirements.txt); it is imported at the top, so a missing install fails loudly.
# Sizes: 15 segments (some classes absent -> tests the zero-division cases) and 500 segments.
# ---------------------------------------------------------------------------

def random_gold_and_pred(n_segments: int, seed: int) -> tuple[dict[str, str], dict[str, str]]:
    """Random gold labels (imbalanced, like MRDA: mostly `other`) and a model that is right ~60% of the time."""
    rng = random.Random(seed)  # fixed seed -> the same "random" data every run, so failures can be reproduced
    weights = [3, 1, 6, 7, 1, 1, 81]  # rough class sizes in percent, in LABELS order (cs, co, aa, bk, ar, cc, other)
    gold, pred = {}, {}
    for i in range(n_segments):
        seg_id = f"seg{i:04d}"  # seg0000, seg0001, ...
        gold[seg_id] = rng.choices(LABELS, weights=weights)[0]  # draw a gold label
        pred[seg_id] = gold[seg_id] if rng.random() < 0.6 else rng.choice(LABELS)  # right 60%, else a random guess
    return gold, pred


@pytest.mark.parametrize("n_segments", [15, 500])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_sklearn(n_segments, seed):
    gold, pred = random_gold_and_pred(n_segments, seed)
    seg_ids = list(gold)  # one fixed order, used to build sklearn's two aligned lists
    y_true = [gold[s] for s in seg_ids]
    y_pred = [pred[s] for s in seg_ids]

    ours = per_class_metrics(as_records(pred), gold)

    # per-class P / R / F1 / support, with sklearn's zero_division=0 to match our rule
    p, r, f1, support = sk.precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), average=None, zero_division=0
    )
    for i, label in enumerate(LABELS):
        assert ours[label]["p"] == pytest.approx(p[i]), label
        assert ours[label]["r"] == pytest.approx(r[i]), label
        assert ours[label]["f1"] == pytest.approx(f1[i]), label
        assert ours[label]["support"] == support[i], label

    # macro-F1 over the 6 targets and over all 7 classes
    assert macro_f1_6(ours) == pytest.approx(
        sk.f1_score(y_true, y_pred, labels=list(TARGET_LABELS), average="macro", zero_division=0)
    )
    assert macro_f1_7(ours) == pytest.approx(
        sk.f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
    )

    # confusion matrix (sklearn also uses rows = gold, columns = predicted)
    ours_counts = confusion_matrix(as_records(pred), gold)
    assert ours_counts == sk.confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()

    # row-normalised matrix (sklearn's normalize="true" also turns empty rows into 0)
    assert_matrix_close(
        row_normalised(ours_counts),
        sk.confusion_matrix(y_true, y_pred, labels=list(LABELS), normalize="true").tolist(),
    )
