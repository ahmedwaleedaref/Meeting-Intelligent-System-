"""Checks for the annotation outputs."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from smi import labels
from smi.data.annotate.links.proposals import AMBIGUOUS, MALFORMED, NO_PAIR, OK
from smi.data.annotate.records import AnnotatedRow, LinkRecord, ParsedRow
from smi.data.annotate.rows import ERROR_FLAGS
from smi.data.annotate.tags import DISRUPTION_MARKERS, PROPOSAL_LABELS

_ROW_KEYS: Final = tuple(AnnotatedRow.__annotations__)
_LINK_KEYS: Final = tuple(LinkRecord.__annotations__)
_STATUSES: Final = frozenset({OK, NO_PAIR, MALFORMED, AMBIGUOUS})


@dataclass
class Findings:
    """Problems as `problem -> count and first example`, so one bug is stated only once, not multiple times."""

    counts: Counter[str] = field(default_factory=Counter)
    first: dict[str, str] = field(default_factory=dict)

    def add(self, problem: str, where: str) -> None:
        self.counts[problem] += 1
        self.first.setdefault(problem, where)

    def messages(self) -> list[str]:
        return [f"{p}: {n} time(s), first at {self.first[p]}" for p, n in self.counts.items()]


def check_contract(
    parsed: Sequence[ParsedRow], annotated: Sequence[AnnotatedRow], links: Sequence[LinkRecord]
) -> list[str]:
    return [*check_rows(parsed, annotated), *check_links(annotated, links)]


def check_rows(parsed: Sequence[ParsedRow], annotated: Sequence[AnnotatedRow]) -> list[str]:
    if len(parsed) != len(annotated):
        return [f"row count changed: {len(parsed)} parsed, {len(annotated)} annotated"]
    findings = Findings()
    for source, row in zip(parsed, annotated):
        for problem in _row_problems(source, row):
            findings.add(problem, row["seg_id"])
    seg_ids = [row["seg_id"] for row in annotated]
    if len(seg_ids) != len(set(seg_ids)):
        findings.add("seg_id is not unique", "file")
    order = [(row["meeting"], row["position"]) for row in annotated]
    if order != sorted(order):
        findings.add("rows are not sorted by (meeting, position)", "file")
    return findings.messages()


def check_links(annotated: Sequence[AnnotatedRow], links: Sequence[LinkRecord]) -> list[str]:
    proposals = [row for row in annotated if row["is_layer2_proposal"]]
    if [link["proposal_id"] for link in links] != [row["seg_id"] for row in proposals]:
        return ["links are not exactly one record per proposal row, in row order"]
    position_of = {row["seg_id"]: row["position"] for row in annotated}
    findings = Findings()
    for row, link in zip(proposals, links):
        for problem in _link_problems(row, link, position_of):
            findings.add(problem, link["proposal_id"])
    return findings.messages()


def _row_problems(source: ParsedRow, row: AnnotatedRow) -> list[str]:
    if tuple(row) != _ROW_KEYS:
        return ["key order differs from AnnotatedRow"]
    targets = row["target_tags"]
    has_error = not ERROR_FLAGS.isdisjoint(row["quality_flags"])
    raw_tags = set(labels.RAW_TAG_TO_LABEL) | set(DISRUPTION_MARKERS)
    checks = (
        (all(row[key] == value for key, value in source.items()), "a parsed field was changed"),
        (
            row["label"] in labels.LABELS and row["label_bk_merged"] in labels.LABELS,
            "a label is outside LABELS",
        ),
        (
            targets == [label for label in labels.TARGET_LABELS if label in targets],
            "target_tags are not distinct and ordered as TARGET_LABELS",
        ),
        (
            (row["label"] in targets) if targets else (row["label"] == labels.LABELS[-1]),
            "label does not follow target_tags",
        ),
        (row["is_multi_target"] == (len(targets) >= 2), "is_multi_target disagrees with target_tags"),
        (
            row["is_layer2_proposal"] == any(label in PROPOSAL_LABELS for label in targets),
            "is_layer2_proposal disagrees with target_tags",
        ),
        (set(row["disruption_markers"]) <= set(DISRUPTION_MARKERS), "unexpected disruption marker"),
        (not set(row["general_tags"]) & raw_tags, "general_tags holds a target tag or marker"),
        (not (row["is_nonspeech"] and row["is_nonlabeled"]), "row is both non-speech and non-labeled"),
        (not row["is_empty"] or row["text"] == "", "is_empty is set on a row that has text"),
        (
            not has_error or not (row["is_nonspeech"] or row["is_nonlabeled"] or row["is_empty"]),
            "an error-flagged row carries a non-speech, non-labeled or empty flag",
        ),
    )
    return [message for ok, message in checks if not ok]


def _link_problems(
    row: AnnotatedRow, link: LinkRecord, position_of: dict[str, int]
) -> list[str]:
    if tuple(link) != _LINK_KEYS:
        return ["key order differs from LinkRecord"]
    chains = link["responder_chains"]
    members = [seg_id for chain in chains for seg_id in chain]
    first_positions = [position_of[chain[0]] for chain in chains if chain and chain[0] in position_of]
    adjacency = row["adjacency_raw"]
    checks = (
        (link["status"] in _STATUSES, "unknown status"),
        (link["has_response"] == bool(chains), "has_response disagrees with responder_chains"),
        (all(chains), "a responder chain is empty"),
        (all(seg_id in position_of for seg_id in members), "a chain holds an unknown seg_id"),
        (len(members) == len(set(members)), "a seg_id appears twice in the chains"),
        (first_positions == sorted(first_positions), "chains are not ordered by first position"),
        (link["status"] != NO_PAIR or not link["has_response"], "a no_pair record has a response"),
        (link["status"] != OK or link["pair_group"] is not None, "an ok record has no pair_group"),
        (
            link["adjacency_tokens"] == (adjacency.split(".") if adjacency is not None else []),
            "adjacency_tokens differ from the row's adjacency_raw",
        ),
    )
    return [message for ok, message in checks if not ok]
