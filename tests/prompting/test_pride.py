"""tests for smi.prompting.pride : the letter prior from rotated options , and dividing it out."""

import pytest

from smi.prompting.calibrate import calibrate
from smi.prompting.pride import average, letter_prior, pride
from smi.prompting.render import to_label_order

LETTER_PULL = [0.30, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10]  # the model likes A and B , whatever they mean
BELIEF = [0.02, 0.01, 0.05, 0.06, 0.01, 0.01, 0.84]       # what it thinks of the 7 options (LABELS order)


def rotations(pull, belief):
    """a segment's scores under every rotation , made as the paper's model says : letter pull × belief in the option."""
    scores = []
    for shift in range(7):
        letter_probs = [pull[j] * belief[(j + shift) % 7] for j in range(7)]
        total = sum(letter_probs)
        scores.append(to_label_order([p / total for p in letter_probs], shift))
    return scores


def test_letter_prior_finds_the_letter_pull_whatever_the_belief():
    assert letter_prior(rotations(LETTER_PULL, BELIEF)) == pytest.approx(LETTER_PULL)


def test_dividing_out_the_prior_leaves_the_belief():
    scores = rotations(LETTER_PULL, BELIEF)
    assert calibrate(scores[0], letter_prior(scores)) == pytest.approx(BELIEF)


def test_a_model_without_letter_pull_has_a_uniform_prior():
    assert letter_prior(rotations([1 / 7] * 7, BELIEF)) == pytest.approx([1 / 7] * 7)


def test_letter_prior_needs_every_rotation():
    with pytest.raises(ValueError, match="all 7 rotations"):
        letter_prior(rotations(LETTER_PULL, BELIEF)[:3])


def test_pride_averages_the_estimation_segments_and_debiases_the_rest():
    rotated = {"a": rotations(LETTER_PULL, BELIEF), "b": rotations(LETTER_PULL, list(reversed(BELIEF)))}
    debiased, prior = pride(rotated, {"a"})
    assert prior == pytest.approx(LETTER_PULL)
    assert debiased["a"] == pytest.approx(average(rotated["a"]))
    assert debiased["b"] == pytest.approx(list(reversed(BELIEF)))
