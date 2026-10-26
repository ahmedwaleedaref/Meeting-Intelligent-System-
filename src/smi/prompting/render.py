"""
turn one dataset item (a target segment and its context) into the text of a prompt.

the prompt files live in configs/prompts/ : a template with three blanks ({options} , {examples} ,
{transcript}) and the option lists (with or without definitions). this module only fills the blanks ;
it does not know about models or tokenizers.
"""

from pathlib import Path

from smi.inputs.dataset import SegmentItem
from smi.labels import LABELS

# one answer letter per class , in LABELS order : A = cs , B = co , ... , G = other
LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
assert len(LETTERS) == len(LABELS)

# the same plain-text markers the encoder uses (scripts/trial_run.py)
OPEN_MARKER = "[T] "
CLOSE_MARKER = " [/T]"

# src/smi/prompting/render.py -> parents[3] is the repository root
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts"


def load_prompt(name: str) -> str:
    """the text of configs/prompts/<name> (e.g. "template_v1.txt") , without trailing newlines."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").rstrip("\n")


def render_transcript(item: SegmentItem) -> str:
    """
    the target's window as text , one segment per line , in speaking order , the target in markers :
        <farthest previous segment>
        ...
        [T] <target> [/T]
        <next segment>
        ...
    item["left"] is closest-first , so it is reversed here (reversed() makes a copy : the item is never changed).
    """
    lines = list(reversed(item["left"]))
    lines.append(OPEN_MARKER + item["target_text"] + CLOSE_MARKER)
    lines.extend(item["right"])
    return "\n".join(lines)


def render_examples(shots: list[SegmentItem]) -> str:
    """
    the few-shot block : each solved example as its transcript plus its answer letter.
    no shots -> "" , so the template's {examples} line simply stays empty (zero-shot).
    """
    if len(shots) == 0:
        return ""
    blocks = []
    for shot in shots:
        answer = LETTERS[shot["label"]]  # label is an index into LABELS , the same order as LETTERS
        blocks.append("Transcript:\n" + render_transcript(shot) + "\nAnswer: " + answer)
    return "\nExamples:\n\n" + "\n\n".join(blocks) + "\n"


def render_prompt(item: SegmentItem, template: str, options: str, shots: list[SegmentItem]) -> str:
    """
    the full prompt for one target segment.
    template : the text of a template file (load_prompt("template_v1.txt"))
    options  : the text of an options file (load_prompt("definitions_v1.txt") or load_prompt("names_v1.txt"))
    shots    : solved examples from train ([] for zero-shot)
    """
    return template.format(
        options=options,
        examples=render_examples(shots),
        transcript=render_transcript(item),
    )
