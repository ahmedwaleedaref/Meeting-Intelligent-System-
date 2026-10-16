"""Helpers shared by the evaluation-harness tests (not a test file: pytest does not collect it).

Imported as `from helpers import ...`: pytest puts tests/eval on sys.path before importing the test files.
"""

import pytest
import sklearn.metrics as sk  # required (requirements.txt): a missing sklearn must fail, not skip

from smi.eval.layer1_metrics import confusion_matrix, macro_f1_6, macro_f1_7, per_class_metrics, row_normalised
from smi.labels import LABELS, TARGET_LABELS


def as_records(predicted: dict[str, str]) -> dict[str, dict]:
    """Turn {seg_id: label} into the shape load_predictions returns: {seg_id: {"label", "scores"}}."""
    return {seg_id: {"label": label, "scores": None} for seg_id, label in predicted.items()}


def assert_matrix_close(actual: list[list[float]], expected: list[list[float]]) -> None:
    """Check two matrices are equal row by row, allowing tiny float rounding differences."""
    assert len(actual) == len(expected)  # same number of rows
    for actual_row, expected_row in zip(actual, expected):  # walk both matrices row by row
        assert actual_row == pytest.approx(expected_row)  # approx: 1/3 written two ways still counts as equal


def assert_matches_sklearn(gold: dict[str, str], pred: dict[str, str]) -> None:
    """Every number the harness produces for (gold, pred) must equal sklearn.metrics with zero_division=0 (T4)."""
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
