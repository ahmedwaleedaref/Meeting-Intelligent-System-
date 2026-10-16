"""Task 3: acceptable_responses and the Layer-2 gold view."""

from pathlib import Path
from typing import Any

import pytest

from release_fixtures import build_release, c2_fixture, link, write_inputs
from smi.data.release import acceptable_responses, load_release

RECORD = {"status": "ok", "responder_chains": [["r1", "r2"], ["r3"]]}


@pytest.mark.parametrize(
    ("responders", "segments", "expected"),
    [
        ("all", "first", {"r1", "r3"}),
        ("all", "any", {"r1", "r2", "r3"}),
        ("first", "first", {"r1"}),
        ("first", "any", {"r1", "r2"}),
    ],
)
def test_all_four_policy_combinations(responders: str, segments: str, expected: set[str]) -> None:
    """Each (responders, segments) combination returns its hand-computed set."""
    assert acceptable_responses(RECORD, responders=responders, segments=segments) == expected


def test_default_policy_is_all_responders_first_segment() -> None:
    """With no options the set holds the first segment of every chain."""
    assert acceptable_responses(RECORD) == acceptable_responses(RECORD, responders="all", segments="first")


def test_acceptable_set_does_not_depend_on_chain_content_order_within_the_set() -> None:
    """The result is a set, so duplicates across chains collapse."""
    record = {"status": "ok", "responder_chains": [["r1"], ["r1", "r2"]]}
    assert acceptable_responses(record) == {"r1"}


def test_no_pair_has_an_empty_acceptable_set() -> None:
    """A no_pair record accepts nothing, which is the only case where none is correct."""
    assert acceptable_responses({"status": "no_pair", "responder_chains": []}) == set()


@pytest.mark.parametrize("status", ["malformed", "ambiguous", "weird", None])
def test_records_outside_ok_and_no_pair_are_not_eligible(status: Any) -> None:
    """Malformed and ambiguous records are never turned into a none target."""
    with pytest.raises(ValueError, match="not eligible"):
        acceptable_responses({"status": status, "responder_chains": []})


@pytest.mark.parametrize(
    "record",
    [
        {"status": "ok", "responder_chains": []},
        {"status": "no_pair", "responder_chains": [["r1"]]},
    ],
)
def test_status_and_chains_must_agree(record: dict[str, Any]) -> None:
    """ok needs at least one chain and no_pair needs none."""
    with pytest.raises(ValueError, match="status and responder_chains disagree"):
        acceptable_responses(record)


@pytest.mark.parametrize("chains", ["r1", [["r1"], "r2"], [[1]], [[""]], None])
def test_chain_structure_is_checked(chains: Any) -> None:
    """Chains must be lists of lists of non-empty seg_id strings."""
    with pytest.raises(ValueError, match="list of lists of non-empty seg_id strings"):
        acceptable_responses({"status": "ok", "responder_chains": chains})


def test_unknown_policy_values_are_rejected() -> None:
    """Only all/first and first/any are accepted."""
    with pytest.raises(ValueError, match="responders must be"):
        acceptable_responses(RECORD, responders="some")
    with pytest.raises(ValueError, match="segments must be"):
        acceptable_responses(RECORD, segments="last")


@pytest.fixture
def release(tmp_path: Path):
    return load_release(build_release(tmp_path)[0])


def test_gold_follows_link_order_and_pairs_each_record_with_its_row(release: Any) -> None:
    """Gold items come in link order with their proposal row and link attached."""
    gold = release.layer2_gold()
    assert [item["proposal_id"] for item in gold] == ["m1-p", "m1-q", "m2-p"]
    for item in gold:
        assert item["proposal"]["seg_id"] == item["proposal_id"] == item["link"]["proposal_id"]


def test_none_target_occurs_only_for_an_empty_set(release: Any) -> None:
    """A gold item has an empty acceptable set exactly when its status is no_pair."""
    for item in release.layer2_gold():
        assert (item["acceptable_responses"] == set()) == (item["link"]["status"] == "no_pair")


def test_eligibility_follows_target_tags_not_the_label(release: Any) -> None:
    """A row labelled cc with cs in its tags is still part of the gold."""
    item = next(item for item in release.layer2_gold() if item["proposal_id"] == "m1-q")
    assert item["proposal"]["label"] == "cc" and "cs" in item["proposal"]["target_tags"]


def test_default_acceptable_sets_match_the_chains(release: Any) -> None:
    """Gold sets use the default policy over the stored chains."""
    sets = {item["proposal_id"]: item["acceptable_responses"] for item in release.layer2_gold()}
    assert sets == {"m1-p": {"m1-r"}, "m1-q": set(), "m2-p": {"m2-r"}}


def test_malformed_and_ambiguous_records_stay_in_the_release_but_out_of_gold(tmp_path: Path) -> None:
    """Records that cannot be resolved are stored for audit and excluded from gold."""
    rows, _, provenance = c2_fixture()
    links = [link("m1-p", [["m1-r"]]), link("m1-q", [], "malformed"), link("m2-p", [], "ambiguous")]
    release = load_release(build_release(tmp_path, inputs=write_inputs(tmp_path / "in", rows, links, provenance))[0])
    assert len(release.links) == 3
    assert [item["proposal_id"] for item in release.layer2_gold()] == ["m1-p"]
    assert release.metadata["counts"]["links_by_status"]["malformed"] == 1
    assert release.metadata["counts"]["links_by_status"]["ambiguous"] == 1


def test_gold_size_matches_the_ok_and_no_pair_counts(release: Any) -> None:
    """The gold size equals ok plus no_pair in the metadata counts."""
    statuses = release.metadata["counts"]["links_by_status"]
    assert len(release.layer2_gold()) == statuses["ok"] + statuses["no_pair"]
