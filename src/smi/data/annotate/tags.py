""" File that contains the tags used for annotation of the data. The tags are used to label the data with specific information that can be used for analysis and processing."""

from smi import labels
import re

OTHER = labels.LABELS[-1]  # "other"

# Order of precedence for the tags. The lower the number, the higher the precedence. This is used to determine which tag to use when multiple tags are present in a single utterance.
LABEL_PRECEDENCE: dict[str, int] = {
    "cc": 0,
    "co": 1,
    "cs": 2,
    "ar": 3,
    "aa": 4,
    "bk": 5,
}

# The tags that are used for annotation of the data. The tags are used to label the data with specific information (suggestion, command) that can be used for analysis and processing.
PROPOSAL_LABELS: tuple[str, ...] = ("cs","co")

DISRUPTION_MARKERS: tuple[str, ...] = ("%-", "%--")

TAG_SEPARATORS = re.compile(r"[\^.]")


def utterance_tokens(part_type: str | None) -> list[str]:
    label = (part_type or "").split(":",1)[0]
    return [token for token in TAG_SEPARATORS.split(label) if token]


def target_tags(tokens: list[str], merge_b: bool = False) -> list[str]:
    classes = {labels.RAW_TAG_TO_LABEL[t] for t in tokens if t in labels.RAW_TAG_TO_LABEL}
    if merge_b and "b" in tokens:
        classes.add("bk")
    return [label for label in labels.TARGET_LABELS if label in classes]


def pick_label(targets: list[str]) -> str:
    ranked = [tag for tag in targets if tag in LABEL_PRECEDENCE]
    return min(ranked, key=LABEL_PRECEDENCE.__getitem__, default=OTHER)


def general_tags(tokens: list[str]) -> list[str]:
    remaining = [t for t in tokens if t not in labels.RAW_TAG_TO_LABEL]
    rem_after_disruption = [t for t in remaining if t not in DISRUPTION_MARKERS]
    return list(dict.fromkeys(rem_after_disruption))


def disruption_markers(tokens: list[str]) -> list[str]:
    return list(dict.fromkeys(t for t in tokens if t in DISRUPTION_MARKERS))
