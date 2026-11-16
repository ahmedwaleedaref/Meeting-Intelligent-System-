"""tests for smi.prompting.render : turning a dataset item into prompt text."""

import copy

import pytest

from smi.labels import LABELS
from smi.prompting.render import (LETTERS, load_prompt, render_prompt, render_transcript, rotate_options,
                                  to_label_order, to_letter_order)


def make_item(target, left=(), right=(), label=0, seg_id="m-1"):
    """a SegmentItem by hand ; left / right are closest-first , like the Dataset's."""
    return {"seg_id": seg_id, "label": label, "left": list(left), "target_text": target, "right": list(right)}


def test_letters_follow_the_label_order():
    # A..G must line up with LABELS , otherwise every answer letter would map to the wrong class
    assert len(LETTERS) == len(LABELS) == 7
    assert dict(zip(LETTERS, LABELS)) == {"A": "cs", "B": "co", "C": "aa", "D": "bk", "E": "ar", "F": "cc", "G": "other"}


def test_k0_transcript_is_the_marked_target_only():
    assert render_transcript(make_item("OK .")) == "[T] OK . [/T]"


def test_context_is_in_speaking_order():
    # left is closest-first : "two" was said just before the target , "one" before that
    item = make_item("target", left=["two", "one"], right=["three", "four"])
    assert render_transcript(item) == "one\ntwo\n[T] target [/T]\nthree\nfour"


def test_rendering_never_changes_the_item():
    item = make_item("target", left=["two", "one"], right=["three"])
    before = copy.deepcopy(item)
    render_transcript(item)
    assert item == before


def test_option_files_list_the_seven_letters_in_order():
    for name in ("definitions_v1.txt", "names_v1.txt"):
        lines = load_prompt(name).split("\n")
        assert [line[:2] for line in lines] == [letter + ")" for letter in LETTERS], name


def test_every_template_has_each_blank_exactly_once():
    # a missing blank would silently drop the options , the examples or the transcript from every prompt
    for name in ("template_v1.txt", "template_p1.txt", "template_p2.txt"):
        template = load_prompt(name)
        for blank in ("{options}", "{examples}", "{transcript}"):
            assert template.count(blank) == 1, (name, blank)


def test_zero_shot_prompt_is_complete():
    prompt = render_prompt(make_item("I'll send it ."), load_prompt("template_v1.txt"), load_prompt("names_v1.txt"), [])
    assert "{" not in prompt  # every blank was filled
    assert "Examples:" not in prompt
    assert "A) suggestion" in prompt and "G) other" in prompt
    assert "[T] I'll send it . [/T]" in prompt
    assert prompt.endswith("Answer with the letter of one option only.")


def test_few_shot_examples_carry_their_answer_letters():
    shots = [make_item("maybe we should try it", label=0), make_item("uhhuh", label=6)]
    prompt = render_prompt(make_item("No ."), load_prompt("template_v1.txt"), load_prompt("names_v1.txt"), shots)
    assert "[T] maybe we should try it [/T]\nAnswer: A" in prompt
    assert "[T] uhhuh [/T]\nAnswer: G" in prompt
    # the examples come before the segment to classify
    assert prompt.index("Answer: G") < prompt.index("[T] No . [/T]")


def test_rotating_by_zero_keeps_the_options_file():
    for name in ("definitions_v1.txt", "names_v1.txt"):
        assert rotate_options(load_prompt(name), 0) == load_prompt(name)


def test_rotating_by_one_moves_every_option_up_one_letter():
    lines = rotate_options(load_prompt("names_v1.txt"), 1).split("\n")
    assert lines[0] == "A) command" and lines[5] == "F) other" and lines[6] == "G) suggestion"


def test_rotation_refuses_options_in_another_form():
    with pytest.raises(ValueError, match="one line per letter"):
        rotate_options("A) suggestion\nB) other", 1)


def test_rotated_examples_answer_with_their_options_new_letter():
    shots = [make_item("maybe we should try it", label=0), make_item("uhhuh", label=6)]
    prompt = render_prompt(make_item("No ."), load_prompt("template_v1.txt"), load_prompt("names_v1.txt"), shots, 1)
    assert "G) suggestion" in prompt and "F) other" in prompt
    assert "[T] maybe we should try it [/T]\nAnswer: G" in prompt
    assert "[T] uhhuh [/T]\nAnswer: F" in prompt


def test_letter_probabilities_go_back_to_their_labels():
    # with shift 1 , letter A shows co : A's probability belongs to co
    letter_probs = [0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05]
    assert to_label_order(letter_probs, 1)[LABELS.index("co")] == 0.30
    for shift in range(len(LETTERS)):
        assert to_letter_order(to_label_order(letter_probs, shift), shift) == letter_probs
