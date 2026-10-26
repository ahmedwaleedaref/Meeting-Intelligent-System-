"""
tests for smi.prompting.backend , on the real Qwen/Qwen2.5-0.5B-Instruct (CPU , fp32).
skipped when the model cannot be loaded (no internet and not in the local Hugging Face cache).
"""

import pytest

from smi.prompting.backend import load_model, score_batch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "Answer with one letter from A to G.",
    "Here is a much longer prompt , so the batch needs padding . " * 5 + "Answer with one letter from A to G.",
    "Which letter comes first in the alphabet ? Answer with one letter.",
]


@pytest.fixture(scope="module")
def loaded():
    try:
        return load_model(MODEL)
    except Exception as error:  # no network , no cache , ...
        pytest.skip(f"cannot load {MODEL}: {error}")


def test_seven_different_single_token_letters(loaded):
    _, _, letter_ids = loaded
    assert len(letter_ids) == 7 and len(set(letter_ids)) == 7


def test_scores_are_probabilities(loaded):
    model, tokenizer, letter_ids = loaded
    for result in score_batch(model, tokenizer, letter_ids, PROMPTS):
        assert len(result["probs"]) == 7
        assert all(0.0 <= p <= 1.0 for p in result["probs"])
        assert abs(sum(result["probs"]) - 1.0) < 1e-5
        assert 0.0 <= result["letter_mass"] <= 1.0


def test_a_batch_gives_the_same_scores_as_one_prompt_at_a_time(loaded):
    # the prompts have different lengths , so the batch is padded : padding must not change any score
    model, tokenizer, letter_ids = loaded
    together = score_batch(model, tokenizer, letter_ids, PROMPTS)
    for prompt, batched in zip(PROMPTS, together):
        alone = score_batch(model, tokenizer, letter_ids, [prompt])[0]
        assert batched["probs"] == pytest.approx(alone["probs"], abs=1e-4)
        assert batched["letter_mass"] == pytest.approx(alone["letter_mass"], abs=1e-4)


def test_an_obvious_question_gets_the_obvious_letter(loaded):
    # a sanity check that we read the right position : "which letter comes first" -> A
    model, tokenizer, letter_ids = loaded
    result = score_batch(model, tokenizer, letter_ids, [PROMPTS[2]])[0]
    assert result["probs"].index(max(result["probs"])) == 0
