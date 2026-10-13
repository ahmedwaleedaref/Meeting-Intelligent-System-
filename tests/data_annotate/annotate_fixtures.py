"""Builders for parsed rows, annotated meetings and link records used by the Task 2 tests."""

from collections import Counter
from collections.abc import Sequence
from typing import Union

from smi.data.annotate.links.groups import build_groups
from smi.data.annotate.links.proposals import build_links
from smi.data.annotate.records import AnnotatedRow, LinkRecord, ParsedRow
from smi.data.annotate.rows import annotate_row

ActSpec = tuple[str, Union[str, None], Union[str, list[str]]]


def parsed_row(
    seg_id: str = "m1-a",
    position: int = 0,
    part_type: str | None = "s",
    *,
    meeting: str = "m1",
    source_da_id: str | None = None,
    adjacency_raw: str | None = None,
    text: str = "word",
    quality_flags: Sequence[str] = (),
    part_index: int = 0,
    n_parts: int = 1,
) -> ParsedRow:
    return {
        "seg_id": seg_id,
        "meeting": meeting,
        "agent": "A",
        "channel": "c1",
        "speaker_id": "me001",
        "src_start": float(position),
        "src_end": float(position) + 1.0,
        "start": float(position),
        "end": float(position) + 1.0,
        "timing_method": "element",
        "position": position,
        "source_da_id": source_da_id or seg_id,
        "element_ordinal": position,
        "part_index": part_index,
        "n_parts": n_parts,
        "raw_type": part_type,
        "original_type": part_type,
        "part_type": part_type,
        "adjacency_raw": adjacency_raw,
        "comment": None,
        "text": text,
        "words": [],
        "quality_flags": list(quality_flags),
    }


def annotated(part_type: str | None, **overrides: object) -> AnnotatedRow:
    return annotate_row(parsed_row(part_type=part_type, **overrides))


def meeting_parsed(specs: Sequence[ActSpec], meeting: str = "m1") -> list[ParsedRow]:
    """Turn (act_id, adjacency, part_type or part_types) specs into parsed rows."""
    rows: list[ParsedRow] = []
    for act_id, adjacency, part_types in specs:
        parts = [part_types] if isinstance(part_types, str) else list(part_types)
        for index, part_type in enumerate(parts):
            seg_id = act_id if len(parts) == 1 else f"{act_id}_p{index + 1}"
            rows.append(
                parsed_row(
                    seg_id,
                    len(rows),
                    part_type,
                    meeting=meeting,
                    source_da_id=act_id,
                    adjacency_raw=adjacency,
                    part_index=index,
                    n_parts=len(parts),
                )
            )
    return rows


def meeting_rows(specs: Sequence[ActSpec], meeting: str = "m1") -> list[AnnotatedRow]:
    return [annotate_row(row) for row in meeting_parsed(specs, meeting)]


def run_links(
    specs: Sequence[ActSpec],
) -> tuple[list[AnnotatedRow], dict[str, LinkRecord], Counter[str]]:
    rows = meeting_rows(specs)
    grouping = build_groups(rows)
    links = build_links(rows, grouping)
    return rows, {link["proposal_id"]: link for link in links}, grouping.notes
