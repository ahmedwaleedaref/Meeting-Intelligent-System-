"""Task 2: tag reading, target tags, label precedence and marker helpers."""

from itertools import combinations

import pytest

from smi.data.annotate.tags import (
    LABEL_PRECEDENCE,
    PROPOSAL_LABELS,
    disruption_markers,
    general_tags,
    pick_label,
    target_tags,
    utterance_tokens,
)


@pytest.mark.parametrize(
    ("part_type", "expected"),
    [
        ("s^cs", ["s", "cs"]),
        ("s.%--", ["s", "%--"]),
        ("s^df.%-", ["s", "df", "%-"]),
        ("s^^cs", ["s", "cs"]),
        ("", []),
        (None, []),
    ],
)
def test_utterance_tokens_split_on_caret_and_dot(part_type: str | None, expected: list[str]) -> None:
    """Tokens are split on ^ and . and empty pieces are dropped."""
    assert utterance_tokens(part_type) == expected


def test_text_after_a_colon_is_not_read() -> None:
    """Only the part before ':' is read, matching the verified multi-target count."""
    assert utterance_tokens("s^cs:s^co") == ["s", "cs"]
    assert utterance_tokens("s^aa:s^cc") == ["s", "aa"]


def test_plus_variants_are_collapsed_into_their_base_class() -> None:
    """aap and arp become aa and ar in target_tags."""
    assert target_tags(["s", "aap"]) == ["aa"]
    assert target_tags(["s", "arp"]) == ["ar"]


def test_target_tags_follow_labels_order_and_are_distinct() -> None:
    """Target tags are distinct and ordered as TARGET_LABELS, not as written."""
    assert target_tags(["ar", "cs", "aa"]) == ["cs", "aa", "ar"]
    assert target_tags(["aap", "aa"]) == ["aa"]
    assert target_tags(["s", "df"]) == []


def test_bare_b_only_counts_when_merging() -> None:
    """A bare b is not a target tag unless the bk-merged variant is requested."""
    assert target_tags(["b"]) == []
    assert target_tags(["b"], merge_b=True) == ["bk"]
    assert target_tags(["bk", "b"], merge_b=True) == ["bk"]


def test_precedence_order_is_cc_co_cs_ar_aa_bk() -> None:
    """The precedence table is the documented D3 order."""
    assert sorted(LABEL_PRECEDENCE, key=LABEL_PRECEDENCE.__getitem__) == ["cc", "co", "cs", "ar", "aa", "bk"]


def test_pick_label_prefers_the_higher_precedence_for_every_pair() -> None:
    """For every pair of target tags the earlier one in the precedence order wins."""
    ordered = sorted(LABEL_PRECEDENCE, key=LABEL_PRECEDENCE.__getitem__)
    for higher, lower in combinations(ordered, 2):
        assert pick_label([lower, higher]) == higher
        assert pick_label([higher, lower]) == higher


def test_pick_label_is_other_without_targets() -> None:
    """No target tag gives the other class."""
    assert pick_label([]) == "other"


def test_pick_label_returns_the_only_tag() -> None:
    """A single target tag is returned as the label."""
    for tag in LABEL_PRECEDENCE:
        assert pick_label([tag]) == tag


def test_proposal_labels_are_cs_and_co() -> None:
    """Layer-2 proposals are the cs and co classes."""
    assert PROPOSAL_LABELS == ("cs", "co")


def test_general_tags_exclude_targets_and_markers_and_stay_distinct() -> None:
    """General tags hold everything but target tags and disruption markers, once each."""
    tokens = ["s", "df", "aa", "df", "%--", "arp", "b"]
    assert general_tags(tokens) == ["s", "df", "b"]


def test_disruption_markers_are_detected_once_each() -> None:
    """Only %- and %-- count as disruption markers and duplicates collapse."""
    assert disruption_markers(["s", "%-", "%--", "%-"]) == ["%-", "%--"]
    assert disruption_markers(["s", "%"]) == []
