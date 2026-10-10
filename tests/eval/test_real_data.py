"""Tests of the harness on the real frozen data (task B3).

Uses the per-meeting files made by scripts/make_meeting_files.py.
They are local only (data/ is not in git), so every test here is SKIPPED, not failed,
when the folder is missing. The toy tests in test_layer1_metrics.py always run.

Run from the repo root:   python -m pytest tests/eval/test_real_data.py -v
"""

from pathlib import Path

import pytest

from smi.eval.gold import load_gold
from smi.eval.layer1_metrics import macro_f1_6, per_class_metrics
from smi.labels import LABELS, TARGET_LABELS

MEETINGS_DIR = Path(__file__).resolve().parents[2] / "data" / "v4_meetings"

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
