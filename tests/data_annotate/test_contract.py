"""Task 2: the C2 contract checks (rows pass through unchanged, links match proposals)."""

import copy
from collections.abc import Callable
from typing import Any

import pytest

from annotate_fixtures import meeting_parsed
from smi.data.annotate.contract import check_contract, check_links, check_rows
from smi.data.annotate.links.groups import build_groups
from smi.data.annotate.links.proposals import build_links
from smi.data.annotate.rows import annotate_row

SPECS = [
    ("p", "1a", "s^cs"),
    ("r1", "1b-1", "s^aa"),
    ("r2", "1b-2", "s"),
    ("x", None, "x"),
    ("q", None, "s^arp^cs"),
]

Case = tuple[list[Any], list[Any], list[Any]]


@pytest.fixture
def case() -> Case:
    parsed = meeting_parsed(SPECS)
    rows = [annotate_row(row) for row in parsed]
    links = build_links(rows, build_groups(rows))
    return parsed, rows, links


def _set(index: int, key: str, value: Any) -> Callable[[Case], None]:
    def mutate(case: Case) -> None:
        case[1][index][key] = value

    return mutate


def _reorder_keys(case: Case) -> None:
    case[1][0] = dict(reversed(list(case[1][0].items())))


def _duplicate_seg_id(case: Case) -> None:
    case[0][1]["seg_id"] = case[1][1]["seg_id"] = case[0][0]["seg_id"]


def _swap_positions(case: Case) -> None:
    case[0][0]["position"], case[0][1]["position"] = 1, 0
    case[1][0]["position"], case[1][1]["position"] = 1, 0


ROW_CASES = [
    ("parsed field changed", _set(0, "text", "changed"), "a parsed field was changed"),
    ("key order", _reorder_keys, "key order differs from AnnotatedRow"),
    ("label outside LABELS", _set(0, "label", "zz"), "a label is outside LABELS"),
    ("unordered target tags", _set(4, "target_tags", ["ar", "cs"]), "target_tags are not distinct and ordered"),
    ("label off target tags", _set(0, "label", "bk"), "label does not follow target_tags"),
    ("multi target flag", _set(4, "is_multi_target", False), "is_multi_target disagrees"),
    ("proposal flag", _set(1, "is_layer2_proposal", True), "is_layer2_proposal disagrees"),
    ("unexpected marker", _set(0, "disruption_markers", ["%"]), "unexpected disruption marker"),
    ("tag leaked into general", _set(0, "general_tags", ["cs"]), "general_tags holds a target tag or marker"),
    ("both drop flags", _set(3, "is_nonlabeled", True), "both non-speech and non-labeled"),
    ("empty with text", _set(0, "is_empty", True), "is_empty is set on a row that has text"),
    ("error row dropped flag", _set(3, "quality_flags", ["ref_unresolved"]), "error-flagged row carries"),
    ("duplicate seg_id", _duplicate_seg_id, "seg_id is not unique"),
    ("unsorted rows", _swap_positions, "rows are not sorted by (meeting, position)"),
]


def test_valid_annotation_passes_the_contract(case: Case) -> None:
    """Rows and links built by the pipeline satisfy every contract check."""
    assert check_contract(*case) == []


@pytest.mark.parametrize(("name", "mutate", "expected"), ROW_CASES, ids=[c[0] for c in ROW_CASES])
def test_row_violations_are_reported(name: str, mutate: Callable[[Case], None], expected: str) -> None:
    """Each broken row rule is reported with its own message."""
    parsed = meeting_parsed(SPECS)
    rows = [annotate_row(row) for row in parsed]
    mutate((parsed, rows, []))
    assert any(expected in problem for problem in check_rows(parsed, rows))


def test_changed_row_count_is_reported(case: Case) -> None:
    """Dropping a row between C1 and C2 is reported as a row-count change."""
    parsed, rows, _ = case
    assert check_rows(parsed, rows[:-1])[0].startswith("row count changed")


def _link(index: int, key: str, value: Any) -> Callable[[list[Any]], None]:
    def mutate(links: list[Any]) -> None:
        links[index][key] = value

    return mutate


LINK_CASES = [
    ("unknown status", _link(0, "status", "weird"), "unknown status"),
    ("has_response off", _link(0, "has_response", False), "has_response disagrees"),
    ("empty chain", _link(0, "responder_chains", [[]]), "a responder chain is empty"),
    ("unknown id", _link(0, "responder_chains", [["ghost"]]), "unknown seg_id"),
    ("repeated id", _link(0, "responder_chains", [["r1"], ["r1"]]), "appears twice"),
    ("unordered chains", _link(0, "responder_chains", [["r2"], ["r1"]]), "not ordered by first position"),
    ("no_pair with response", _link(0, "status", "no_pair"), "a no_pair record has a response"),
    ("ok without group", _link(0, "pair_group", None), "an ok record has no pair_group"),
    ("tokens differ", _link(0, "adjacency_tokens", ["9a"]), "adjacency_tokens differ"),
]


@pytest.mark.parametrize(("name", "mutate", "expected"), LINK_CASES, ids=[c[0] for c in LINK_CASES])
def test_link_violations_are_reported(case: Case, name: str, mutate: Callable[[list[Any]], None], expected: str) -> None:
    """Each broken link rule is reported with its own message."""
    _, rows, links = case
    links = copy.deepcopy(links)
    mutate(links)
    assert any(expected in problem for problem in check_links(rows, links))


def test_link_key_order_is_checked(case: Case) -> None:
    """A link record with reordered keys is reported."""
    _, rows, links = case
    links = copy.deepcopy(links)
    links[0] = dict(reversed(list(links[0].items())))
    assert any("key order differs from LinkRecord" in problem for problem in check_links(rows, links))


def test_missing_link_is_reported(case: Case) -> None:
    """A proposal without a link record fails the one-record-per-proposal rule."""
    _, rows, links = case
    assert check_links(rows, links[:-1]) == ["links are not exactly one record per proposal row, in row order"]


def test_reordered_links_are_reported(case: Case) -> None:
    """Links must follow row order, so a swap is reported."""
    _, rows, links = case
    assert check_links(rows, list(reversed(links))) == [
        "links are not exactly one record per proposal row, in row order"
    ]


def test_link_for_a_non_proposal_is_reported(case: Case) -> None:
    """An extra link for a row that is not a proposal is reported."""
    _, rows, links = case
    extra = copy.deepcopy(links[0])
    extra["proposal_id"] = "r1"
    assert check_links(rows, [*links, extra]) != []
