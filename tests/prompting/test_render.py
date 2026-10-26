"""tests for smi.prompting.render : turning a dataset item into prompt text."""

import copy

from smi.labels import LABELS
from smi.prompting.render import LETTERS, load_prompt, render_prompt, render_transcript


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
