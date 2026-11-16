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


def render_examples(shots: list[SegmentItem], shift: int = 0) -> str:
    """
    the few-shot block : each solved example as its transcript plus its answer letter.
    no shots -> "" , so the template's {examples} line simply stays empty (zero-shot).
    shift : the options' rotation (rotate_options) , so each example answers with its option's current letter
    """
    if len(shots) == 0:
        return ""
    blocks = []
    for shot in shots:
        answer = LETTERS[letter_index(shot["label"], shift)]
        blocks.append("Transcript:\n" + render_transcript(shot) + "\nAnswer: " + answer)
    return "\nExamples:\n\n" + "\n\n".join(blocks) + "\n"


def render_prompt(item: SegmentItem, template: str, options: str, shots: list[SegmentItem], shift: int = 0) -> str:
    """
    the full prompt for one target segment.
    template : the text of a template file (load_prompt("template_v1.txt"))
    options  : the text of an options file (load_prompt("definitions_v1.txt") or load_prompt("names_v1.txt"))
    shots    : solved examples from train ([] for zero-shot)
    shift    : rotate the options by this many letters (rotate_options) ; 0 = the file's order
    """
    return template.format(
        options=rotate_options(options, shift),
        examples=render_examples(shots, shift),
        transcript=render_transcript(item),
    )


# ------------------------------------------------------------------------------
# rotated options : the same 7 options behind other letters (PriDe , pride.py)
# ------------------------------------------------------------------------------
# with shift s , letter j shows the option of class (j + s) % 7 : the letters stay A..G in order , the option
# texts move up s places (a cyclic permutation). shift 1 : A) command , ... , F) other , G) suggestion.


def rotate_options(options: str, shift: int) -> str:
    """
    the options with their texts rotated by shift letters.
    options : the text of an options file , one line per letter , "A) ..." to "G) ..." in order
    raises ValueError if the options are not in that form (the rotation would then mislabel them)
    """
    lines = options.split("\n")
    if [line[:3] for line in lines] != [letter + ") " for letter in LETTERS]:
        raise ValueError("options must be one line per letter, 'A) ...' to 'G) ...', in order")
    texts = [line[3:] for line in lines]
    return "\n".join(f"{letter}) {texts[(j + shift) % len(LETTERS)]}" for j, letter in enumerate(LETTERS))


def letter_index(label: int, shift: int) -> int:
    """the position in LETTERS of class label's letter , with the options rotated by shift."""
    return (label - shift) % len(LETTERS)


def to_label_order(letter_probs: list[float], shift: int) -> list[float]:
    """a rotated prompt's 7 letter probabilities (LETTERS order) , reordered to LABELS order."""
    return [letter_probs[letter_index(label, shift)] for label in range(len(LABELS))]


def to_letter_order(label_probs: list[float], shift: int) -> list[float]:
    """the opposite of to_label_order : probabilities in LABELS order , back to the letters they were read from."""
    return [label_probs[(j + shift) % len(LETTERS)] for j in range(len(LETTERS))]
