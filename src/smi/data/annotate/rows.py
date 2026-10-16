"""Append the annotation fields to one parsed row."""

from typing import Final

from smi.data.annotate.records import AnnotatedRow, ParsedRow
from smi.data.annotate.tags import (
    PROPOSAL_LABELS,
    disruption_markers,
    general_tags,
    pick_label,
    target_tags,
    utterance_tokens,
)

#A row carrying one is a parse problem, so it is not flagged as non-speech, non-labeled or empty.
ERROR_FLAGS: Final[frozenset[str]] = frozenset(
    {"ref_unresolved", "ref_malformed", "time_invalid", "type_unavailable"}
)


def annotate_row(row: ParsedRow) -> AnnotatedRow:
    tokens = utterance_tokens(row["part_type"])
    targets = target_tags(tokens)
    clean = ERROR_FLAGS.isdisjoint(row["quality_flags"])
    return {
        **row,
        "label": pick_label(targets),
        "label_bk_merged": pick_label(target_tags(tokens, merge_b=True)),
        "target_tags": targets,
        "is_multi_target": len(targets) >= 2,
        "is_layer2_proposal": any(label in PROPOSAL_LABELS for label in targets),
        "general_tags": general_tags(tokens),
        "disruption_markers": disruption_markers(tokens),
        "is_nonspeech": clean and tokens == ["x"],
        "is_nonlabeled": clean and tokens == ["z"],
        "is_empty": clean and not row["text"],
    }
