"""Task 2: annotate_row output (labels, tag sets, flags and analysis-only fields)."""

import pytest

from annotate_fixtures import annotated, parsed_row
from smi.data.annotate.records import ANNOTATION_FIELDS
from smi.data.annotate.rows import ERROR_FLAGS, annotate_row
from smi.data.nxt.validation import ERROR_FLAGS as PARSER_ERROR_FLAGS

MULTI_TARGET_CASES = [
    ("s^arp^cs", ["cs", "ar"], "cs"),
    ("s^aap^cs", ["cs", "aa"], "cs"),
    ("s^ar^co", ["co", "ar"], "co"),
    ("s^aa^bk", ["aa", "bk"], "aa"),
    ("s^aa^cc", ["aa", "cc"], "cc"),
    ("s^cs^cc", ["cs", "cc"], "cc"),
]


@pytest.mark.parametrize(("part_type", "label"), [("s^aap", "aa"), ("s^arp", "ar")])
def test_plus_variants_become_their_base_label(part_type: str, label: str) -> None:
    """aap and arp rows are labelled aa and ar."""
    row = annotated(part_type)
    assert row["label"] == label
    assert row["target_tags"] == [label]


@pytest.mark.parametrize(("part_type", "targets", "label"), MULTI_TARGET_CASES)
def test_multi_target_rows_keep_the_full_set_and_follow_precedence(
    part_type: str, targets: list[str], label: str
) -> None:
    """Every observed multi-target pattern keeps all tags and picks the D3 label."""
    row = annotated(part_type)
    assert row["target_tags"] == targets
    assert row["label"] == label
    assert row["is_multi_target"] is True


def test_text_after_a_colon_does_not_add_targets() -> None:
    """For s^cs:s^co only the part before the colon is read."""
    row = annotated("s^cs:s^co")
    assert row["target_tags"] == ["cs"]
    assert row["label"] == "cs"
    assert row["is_multi_target"] is False


@pytest.mark.parametrize("part_type", ["s", "s^cs", "s^bk", None])
def test_single_or_no_target_rows_are_not_multi_target(part_type: str | None) -> None:
    """Rows with zero or one target tag are not multi-target."""
    assert annotated(part_type)["is_multi_target"] is False


def test_row_without_target_tags_is_other() -> None:
    """A row with no target tag is labelled other with an empty tag set."""
    row = annotated("s^df")
    assert (row["label"], row["target_tags"]) == ("other", [])


@pytest.mark.parametrize(
    ("part_type", "expected"),
    [
        ("s^arp^cs", True),
        ("s^ar^co", True),
        ("s^cs", True),
        ("s^co", True),
        ("s^cs^cc", True),
        ("s^aa^cc", False),
        ("s^aa", False),
        ("s^bk", False),
        ("s", False),
    ],
)
def test_layer2_eligibility_follows_target_tags_not_the_label(part_type: str, expected: bool) -> None:
    """A row is a Layer-2 proposal iff its tag set holds cs or co, whatever its label."""
    assert annotated(part_type)["is_layer2_proposal"] is expected


def test_multi_target_proposal_stays_eligible_when_label_is_not_cs_or_co() -> None:
    """A row labelled cc with cs in its tag set is still a proposal."""
    row = annotated("s^cs^cc")
    assert row["label"] == "cc"
    assert row["is_layer2_proposal"] is True


@pytest.mark.parametrize(
    ("part_type", "label", "merged"),
    [
        ("b", "other", "bk"),
        ("s^b", "other", "bk"),
        ("bk", "bk", "bk"),
        ("s^aa^b", "aa", "aa"),
        ("s^b^cs", "cs", "cs"),
        ("s^df", "other", "other"),
    ],
)
def test_bare_b_stays_other_but_is_bk_in_the_merged_variant(part_type: str, label: str, merged: str) -> None:
    """The primary label keeps bare b in other while label_bk_merged counts it as bk."""
    row = annotated(part_type)
    assert (row["label"], row["label_bk_merged"]) == (label, merged)


def test_merged_variant_does_not_change_target_tags() -> None:
    """Only label_bk_merged sees the b tag; target_tags stay unchanged."""
    assert annotated("s^b")["target_tags"] == []


def test_x_row_is_non_speech() -> None:
    """A row whose only tag is x is non-speech."""
    row = annotated("x")
    assert (row["is_nonspeech"], row["is_nonlabeled"]) == (True, False)


def test_z_row_is_non_labeled() -> None:
    """A row whose only tag is z is non-labeled and not non-speech."""
    row = annotated("z")
    assert (row["is_nonspeech"], row["is_nonlabeled"]) == (False, True)


def test_ordinary_row_has_no_drop_flag() -> None:
    """A row with text and an ordinary tag carries no drop flag."""
    row = annotated("s^cs", text="hello")
    assert (row["is_nonspeech"], row["is_nonlabeled"], row["is_empty"]) == (False, False, False)


def test_row_without_words_is_empty() -> None:
    """A row with no words and only the informational no_word_ref flag is empty."""
    row = annotated("s", text="", quality_flags=["no_word_ref"])
    assert row["is_empty"] is True


@pytest.mark.parametrize("error_flag", sorted(ERROR_FLAGS))
@pytest.mark.parametrize("part_type", ["x", "z", "s"])
def test_error_flagged_rows_are_never_non_speech_non_labeled_or_empty(
    error_flag: str, part_type: str
) -> None:
    """A parse-error row passes through without any drop flag."""
    row = annotated(part_type, text="", quality_flags=[error_flag])
    assert (row["is_nonspeech"], row["is_nonlabeled"], row["is_empty"]) == (False, False, False)


def test_error_flag_set_matches_the_parser_vocabulary() -> None:
    """Task 2 treats exactly the parser's error-class flags as parse errors."""
    assert ERROR_FLAGS == frozenset(PARSER_ERROR_FLAGS)


def test_informational_flags_do_not_block_the_drop_flags() -> None:
    """Informational quality flags leave the drop flags active."""
    row = annotated("x", text="", quality_flags=["word_time_missing", "no_word_ref"])
    assert (row["is_nonspeech"], row["is_empty"]) == (True, True)


@pytest.mark.parametrize(
    ("part_type", "markers", "general"),
    [
        ("s.%--", ["%--"], ["s"]),
        ("s.%-", ["%-"], ["s"]),
        ("%--", ["%--"], []),
        ("s^df.%-", ["%-"], ["s", "df"]),
        ("s^df", [], ["s", "df"]),
    ],
)
def test_disruption_markers_are_separated_from_general_tags(
    part_type: str, markers: list[str], general: list[str]
) -> None:
    """%- and %-- land in disruption_markers and never in general_tags."""
    row = annotated(part_type)
    assert row["disruption_markers"] == markers
    assert row["general_tags"] == general


def test_general_tags_never_hold_target_tags_or_markers() -> None:
    """Target tags and markers are removed from the analysis-only general tags."""
    row = annotated("s^df^aa.%--")
    assert row["general_tags"] == ["s", "df"]
    assert row["target_tags"] == ["aa"]


def test_missing_part_type_is_a_plain_other_row() -> None:
    """A row with no part type is other with no tags and no flags."""
    row = annotated(None)
    assert (row["label"], row["target_tags"], row["general_tags"]) == ("other", [], [])


def test_parsed_fields_pass_through_unchanged_and_in_order() -> None:
    """All C1 fields keep their values and order, with annotation fields appended."""
    source = parsed_row("m1-x", 4, "s^cs", adjacency_raw="1a", text="hi")
    row = annotate_row(source)
    assert list(row)[: len(source)] == list(source)
    assert all(row[key] == value for key, value in source.items())
    assert tuple(list(row)[len(source):]) == ANNOTATION_FIELDS


def test_annotate_row_does_not_mutate_its_input() -> None:
    """annotate_row returns a new mapping and leaves the parsed row untouched."""
    source = parsed_row("m1-x", 0, "s^cs")
    snapshot = dict(source)
    annotate_row(source)
    assert source == snapshot
