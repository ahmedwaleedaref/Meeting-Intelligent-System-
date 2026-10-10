"""Tests of the harness on the real frozen data (task B3).

Uses the per-meeting files made by scripts/make_meeting_files.py.
They are local only (data/ is not in git), so every test here is SKIPPED, not failed,
when the folder is missing. The toy tests in test_layer1_metrics.py always run.

Run from the repo root:   python -m pytest tests/eval/test_real_data.py -v
"""

from pathlib import Path

import pytest

from smi.eval.gold import load_gold
from smi.eval.layer1_metrics import macro_f1_6, macro_f1_7, per_class_metrics
from smi.eval.prediction_schema import load_predictions
from smi.labels import LABELS, TARGET_LABELS

MEETINGS_DIR = Path(__file__).resolve().parents[2] / "data" / "v4_meetings"
TOY_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "toy"

needs_data = pytest.mark.skipif(not MEETINGS_DIR.is_dir(), reason=f"{MEETINGS_DIR} not available locally")


def as_records(labels: dict[str, str]) -> dict[str, dict]:
    """Test shortcut: wrap {seg_id: label} as prediction records {seg_id: {"label": label, "scores": None}}."""
    return {seg_id: {"label": label, "scores": None} for seg_id, label in labels.items()}


# ---------------------------------------------------------------------------
# T1: perfect predictions -> every F1 = 1.0
# Gold is used as the predictions. A class with no gold in the split would get 0
# (zero-support decision), so only classes that occur are expected to be 1.0.
# Never run on test (PROTOCOL §7).
# ---------------------------------------------------------------------------

@needs_data
@pytest.mark.parametrize("split", ["train", "val"])
def test_perfect_predictions_on_real_split(split):
    gold = load_gold(MEETINGS_DIR / split)
    metrics = per_class_metrics(as_records(gold), gold)

    for label in LABELS:
        if metrics[label]["support"] == 0:
            assert metrics[label]["f1"] == 0.0, label
        else:
            assert metrics[label]["p"] == 1.0, label
            assert metrics[label]["r"] == 1.0, label
            assert metrics[label]["f1"] == 1.0, label
    assert sum(metrics[label]["support"] for label in LABELS) == len(gold)  # no segment lost


# ---------------------------------------------------------------------------
# T2: everything predicted as `other` -> macro-F1 (6) = 0.0
# ---------------------------------------------------------------------------

@needs_data
@pytest.mark.parametrize("split", ["train", "val"])
def test_all_other_on_real_split(split):
    gold = load_gold(MEETINGS_DIR / split)
    all_other = {seg_id: "other" for seg_id in gold}  # a model that always says "other"
    metrics = per_class_metrics(as_records(all_other), gold)

    for label in TARGET_LABELS:
        assert metrics[label]["p"] == 0.0, label
        assert metrics[label]["r"] == 0.0, label
        assert metrics[label]["f1"] == 0.0, label
    assert macro_f1_6(metrics) == 0.0


# ---------------------------------------------------------------------------
# T3: oracle on a toy set of real segments (toy file + oracle merged)
# 3 val meetings x 10 consecutive segments, 5 labels changed by hand.
# Edits and the hand calculation: tests/fixtures/toy/README.md
# The fixtures are in git, so this test always runs (no skip).
# ---------------------------------------------------------------------------

TOY_EXPECTED = {
    "cs": {"p": 7 / 8, "r": 7 / 8, "f1": 7 / 8, "support": 8},
    "co": {"p": 3 / 4, "r": 1.0, "f1": 6 / 7, "support": 3},
    "aa": {"p": 1 / 2, "r": 1.0, "f1": 2 / 3, "support": 2},
    "bk": {"p": 6 / 7, "r": 6 / 7, "f1": 6 / 7, "support": 7},
    "ar": {"p": 0.0, "r": 0.0, "f1": 0.0, "support": 1},  # never predicted -> zero division
    "cc": {"p": 1.0, "r": 1 / 2, "f1": 2 / 3, "support": 2},
    "other": {"p": 1.0, "r": 6 / 7, "f1": 12 / 13, "support": 7},
}
TOY_MACRO_F1_6 = 8567 / 13104  # (7/8 + 6/7 + 2/3 + 6/7 + 0 + 2/3) / 6
TOY_MACRO_F1_7 = 10583 / 15288  # (7/8 + 6/7 + 2/3 + 6/7 + 0 + 2/3 + 12/13) / 7


def test_toy_oracle():
    gold = load_gold(TOY_DIR / "gold")  # 3 meeting files -> 30 segments
    records = load_predictions(TOY_DIR / "predictions.jsonl")
    metrics = per_class_metrics(records, gold)

    for label in LABELS:
        assert metrics[label] == pytest.approx(TOY_EXPECTED[label]), label  # on failure, the class name is shown
    assert macro_f1_6(metrics) == pytest.approx(TOY_MACRO_F1_6)
    assert macro_f1_7(metrics) == pytest.approx(TOY_MACRO_F1_7)
