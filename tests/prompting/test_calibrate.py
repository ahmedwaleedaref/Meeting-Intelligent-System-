"""tests for smi.prompting.calibrate : dividing out the letter bias."""

import pytest

from smi.prompting.calibrate import calibrate


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
