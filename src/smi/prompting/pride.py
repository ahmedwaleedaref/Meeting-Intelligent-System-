"""
PriDe , debiasing with prior estimation (Zheng et al. 2024 , "Large Language Models Are Not Robust Multiple Choice
Selectors" , ICLR 2024 , https://arxiv.org/abs/2309.03882).

a prompted model may prefer some answer letters (A..G) whatever option they stand for. scoring a segment under the
7 rotations of the options (render.rotate_options) shows every letter with every option once : what a letter keeps
across the rotations is its own pull , the letter prior. dividing that prior out of the usual (unrotated) scores
leaves the preference for the options themselves , e.g. the lean to "other" , which contextual calibration
(calibrate.py) removed together with any letter preference.
"""

import math

from smi.prompting.calibrate import calibrate
from smi.prompting.render import LETTERS, to_letter_order

# a probability that underflowed to 0 in fp32 counts as this , so its log stays finite
SMALLEST_PROB = 1e-30


def letter_prior(rotated: list[list[float]]) -> list[float]:
    """
    one segment's letter prior (the paper's equation 7) : a softmax over the letters of the mean log probability
    each letter got across the rotations.
    rotated : the segment's 7 probabilities (LABELS order) under each rotation , shift 0 , 1 , ... , 6
    returns 7 probabilities in LETTERS order
    raises ValueError unless there is one list per rotation
    """
    if len(rotated) != len(LETTERS):
        raise ValueError(f"expected the scores of all {len(LETTERS)} rotations, got {len(rotated)}")
    letter_probs = [to_letter_order(probs, shift) for shift, probs in enumerate(rotated)]
    mean_log = [sum(math.log(max(probs[j], SMALLEST_PROB)) for probs in letter_probs) / len(rotated)
                for j in range(len(LETTERS))]
    top = max(mean_log)  # subtracted before exp : no overflow
    weights = [math.exp(m - top) for m in mean_log]
    total = sum(weights)
    return [w / total for w in weights]


def average(vectors: list[list[float]]) -> list[float]:
    """the mean of several probability lists , position by position."""
    return [sum(values) / len(vectors) for values in zip(*vectors)]


def pride(rotated: dict[str, list[list[float]]], estimation: set[str]) -> tuple[dict[str, list[float]], list[float]]:
    """
    the paper's algorithm 1. the estimation segments are scored under every rotation : they get the average of
    their rotations (equation 1) and give their letter priors. every other segment only needs its unrotated scores ,
    divided by the mean of those priors and renormalised (equation 8 , calibrate()'s arithmetic ; with shift 0 the
    letters are in LABELS order).
    rotated : seg_id -> its probabilities (LABELS order) under each rotation , shift 0 first
    estimation : the seg_ids the prior is estimated on (no labels are used)
    returns (seg_id -> debiased probabilities in LABELS order , the prior in LETTERS order)
    """
    prior = average([letter_prior(rotated[seg_id]) for seg_id in estimation])
    debiased = {seg_id: average(scores) if seg_id in estimation else calibrate(scores[0], prior)
                for seg_id, scores in rotated.items()}
    return debiased, prior
