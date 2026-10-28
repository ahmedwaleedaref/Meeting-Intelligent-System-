"""
tests for smi.prompting.backend , on the real Qwen/Qwen2.5-0.5B-Instruct (CPU , fp32).
skipped when the model cannot be loaded (no internet and not in the local Hugging Face cache).
"""

import pytest

from smi.prompting.backend import (build_prefix_cache, common_prefix_len, compare_prefix_cache, largest_diff, load_model,
                                   score_batch, score_suffixes, tokenize_prompts)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "Answer with one letter from A to G.",
    "Here is a much longer prompt , so the batch needs padding . " * 5 + "Answer with one letter from A to G.",
    "Which letter comes first in the alphabet ? Answer with one letter.",
]

SHARED_PROMPTS = [
    "Options: A, B, C, D, E, F, G. Here is a shared instruction. " * 4 + ending
    for ending in ("Answer: which letter first?", "Answer with one letter, please, any letter.", "Go.")
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


def test_common_prefix_len_leaves_one_token_of_each_sequence():
    assert common_prefix_len([[1, 2, 3, 4], [1, 2, 5]]) == 2
    assert common_prefix_len([[1, 2, 3], [1, 2, 3, 4]]) == 2
    assert common_prefix_len([[7, 8], [9, 8]]) == 0


def test_prefix_cached_scores_equal_full_prompt_scores(loaded):
    model, tokenizer, letter_ids = loaded
    prompts = SHARED_PROMPTS
    ids = tokenize_prompts(tokenizer, prompts)
    prefix_len = common_prefix_len(ids)
    cache = build_prefix_cache(model, ids[0][:prefix_len], len(prompts))
    cached = score_suffixes(model, letter_ids, cache, prefix_len, [s[prefix_len:] for s in ids], tokenizer.pad_token_id)
    for prompt, result in zip(prompts, cached):
        assert result["probs"] == pytest.approx(score_batch(model, tokenizer, letter_ids, [prompt])[0]["probs"], abs=1e-4)


def test_compare_prefix_cache_passes_in_fp32(loaded):
    model, tokenizer, letter_ids = loaded
    prefix_len = common_prefix_len(tokenize_prompts(tokenizer, SHARED_PROMPTS))
    check = compare_prefix_cache(model, tokenizer, letter_ids, SHARED_PROMPTS, prefix_len)
    assert check["cached"] < 1e-4 and check["cached"] <= check["allowed"]


def test_largest_diff_is_over_every_prompt_and_letter():
    first = [{"probs": [0.5, 0.5]}, {"probs": [0.9, 0.1]}]
    second = [{"probs": [0.4, 0.6]}, {"probs": [0.6, 0.4]}]
    assert largest_diff(first, second) == pytest.approx(0.3)
