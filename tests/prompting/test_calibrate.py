"""tests for smi.prompting.calibrate : dividing out the letter bias."""

import json

import pytest

from smi.prompting.calibrate import calibrate, label_frequencies
from tests.prompting.test_shots import write_split


def test_a_uniform_bias_changes_nothing():
    probs = [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert calibrate(probs, [1 / 7] * 7) == pytest.approx(probs)


def test_result_is_a_probability_distribution():
    result = calibrate([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1], [0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
    assert sum(result) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in result)


def test_a_letter_the_model_always_likes_loses_its_advantage():
    # the model gives A 0.40 on a real input , but it gives A 0.40 to an empty input too : no evidence for A.
    # G gets 0.20 against a usual 0.05 : four times its usual level , so G wins after calibration.
    bias = [0.40, 0.10, 0.10, 0.10, 0.10, 0.15, 0.05]
    probs = [0.40, 0.10, 0.10, 0.10, 0.05, 0.05, 0.20]
    result = calibrate(probs, bias)
    assert result.index(max(result)) == 6


def test_no_target_and_a_uniform_target_give_the_same_result():
    probs, bias = [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1], [0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
    assert calibrate(probs, bias, [1 / 7] * 7) == pytest.approx(calibrate(probs, bias))


def test_a_content_free_input_is_calibrated_to_the_target():
    bias, target = [0.40, 0.10, 0.10, 0.10, 0.10, 0.15, 0.05], [0.02, 0.01, 0.05, 0.06, 0.01, 0.01, 0.84]
    assert calibrate(bias, bias, target) == pytest.approx(target)


def test_label_frequencies_follow_the_train_counts(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    labels = ["other"] * 5 + ["cs", "aa", "aa"]
    with open(train / "M0.jsonl", "w", encoding="utf-8") as f:
        for i, label in enumerate(labels):
            f.write(json.dumps({"seg_id": f"M0-{i}", "position": i, "speaker": "a", "text": "t", "label": label}) + "\n")
    assert label_frequencies(train) == pytest.approx([1 / 8, 0, 2 / 8, 0, 0, 0, 5 / 8])


def test_label_frequencies_refuse_a_folder_that_is_not_train(tmp_path):
    with pytest.raises(ValueError, match="train split"):
        label_frequencies(write_split(tmp_path / "val"))
