"""Typed shapes of parsed rows and annotated rows (key order = schema order)."""

from typing import Final, TypedDict


class Word(TypedDict):
    text: str
    kind: str
    start: float | None
    end: float | None


class ParsedRow(TypedDict):

    seg_id: str
    meeting: str
    agent: str
    channel: str
    speaker_id: str
    src_start: float
    src_end: float
    start: float
    end: float
    timing_method: str
    position: int
    source_da_id: str
    element_ordinal: int
    part_index: int
    n_parts: int
    raw_type: str | None
    original_type: str | None
    part_type: str | None
    adjacency_raw: str | None
    comment: str | None
    text: str
    words: list[Word]
    quality_flags: list[str]


class AnnotatedRow(ParsedRow):

    label: str
    label_bk_merged: str
    target_tags: list[str]
    is_multi_target: bool
    is_layer2_proposal: bool
    general_tags: list[str]
    disruption_markers: list[str]
    is_nonspeech: bool
    is_nonlabeled: bool
    is_empty: bool


ANNOTATION_FIELDS: Final[tuple[str, ...]] = tuple(AnnotatedRow.__annotations__)[
    len(ParsedRow.__annotations__) :
]


class LinkRecord(TypedDict):
    
    proposal_id: str
    responder_chains: list[list[str]]
    has_response: bool
    status: str
    adjacency_tokens: list[str]
    pair_group: str | None
