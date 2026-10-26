"""
end-to-end test of scripts/trial_prompting.py on a tiny fake dataset with the real Qwen 0.5B (CPU).
skipped when the model cannot be loaded.
"""

import json
import sys
from pathlib import Path

import pytest

from smi.eval.prediction_schema import load_predictions
from smi.labels import LABELS
from tests.prompting.test_backend import MODEL
from tests.prompting.test_shots import write_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import trial_prompting  # noqa: E402


@pytest.fixture(scope="module")
def model_available():
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(MODEL)
    except Exception as error:
        pytest.skip(f"cannot load {MODEL}: {error}")


def test_a_few_shot_run_with_context_writes_valid_predictions_and_metrics(tmp_path, model_available):
    data_root = tmp_path / "data"
    write_split(data_root / "train", n_meetings=1, per_label=2)
    write_split(data_root / "val", n_meetings=1, per_label=1)  # 7 segments , one of each class
    (data_root / "source.json").write_text(json.dumps({"data_version": "toy"}), encoding="utf-8")

    trial_prompting.main([
        "--data-root", str(data_root), "--run-name", "toy", "--out-dir", str(tmp_path / "runs"),
        "--past-k", "1", "--future-k", "1", "--shots-per-class", "1", "--batch-size", "4",
    ])

    out = tmp_path / "runs" / "toy"
    records = load_predictions(out / "predictions.jsonl")  # the harness's own checks
    assert len(records) == len(LABELS)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["data_version"] == "toy" and metrics["n_segments"] == len(LABELS)
    assert 0.0 <= metrics["prompting"]["mean_letter_mass"] <= 1.0
    settings = json.loads((out / "settings.json").read_text(encoding="utf-8"))
    assert len(settings["shot_seg_ids"]) == len(LABELS)  # one example per class
