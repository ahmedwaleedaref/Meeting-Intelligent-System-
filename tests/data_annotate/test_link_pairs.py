"""Task 2: adjacency-pair groups, responder chains and link statuses."""

import pytest

from annotate_fixtures import run_links
from smi.data.annotate.contract import check_links


def test_simple_pair_links_proposal_to_its_response() -> None:
    """A cs act followed by a b response gives one ok record with one chain."""
    rows, links, _ = run_links([("p", "1a", "s^cs"), ("r", "1b", "s^aa")])
    link = links["p"]
    assert link["responder_chains"] == [["r"]]
    assert (link["status"], link["has_response"]) == ("ok", True)
    assert link["adjacency_tokens"] == ["1a"]
    assert link["pair_group"] == "m1:1@0"
    assert check_links(rows, list(links.values())) == []


def test_multi_responder_pair_gives_one_chain_per_responder() -> None:
    """b-1 and b-2 responses make two chains."""
    _, links, _ = run_links([("p", "1a", "s^cs"), ("r1", "1b-1", "s^aa"), ("r2", "1b-2", "s^bk")])
    assert links["p"]["responder_chains"] == [["r1"], ["r2"]]
    assert links["p"]["status"] == "ok"


def test_chains_are_ordered_by_first_response_position_not_speaker_number() -> None:
    """The responder that speaks first comes first, whatever its speaker number."""
    _, links, _ = run_links([("p", "1a", "s^cs"), ("r2", "1b-2", "s"), ("r1", "1b-1", "s")])
    assert links["p"]["responder_chains"] == [["r2"], ["r1"]]


def test_continuation_extends_the_responders_chain() -> None:
    """A later b+ from the same responder extends that chain."""
    _, links, _ = run_links(
        [("p", "1a", "s^cs"), ("r1", "1b", "s"), ("other", None, "s"), ("r2", "1b+", "s")]
    )
    assert links["p"]["responder_chains"] == [["r1", "r2"]]


def test_bare_continuation_extends_the_only_numbered_chain() -> None:
    """A bare b+ extends the single existing chain even if that chain is numbered."""
    _, links, _ = run_links([("p", "1a", "s^cs"), ("r1", "1b-1", "s"), ("r2", "1b+", "s")])
    assert links["p"]["responder_chains"] == [["r1", "r2"]]


def test_bare_continuation_with_several_chains_is_ambiguous() -> None:
    """A bare b+ with two possible chains is not attached and marks the pair ambiguous."""
    _, links, notes = run_links(
        [("p", "1a", "s^cs"), ("r1", "1b-1", "s"), ("r2", "1b-2", "s"), ("r3", "1b+", "s")]
    )
    assert links["p"]["status"] == "ambiguous"
    assert links["p"]["responder_chains"] == [["r1"], ["r2"]]
    assert notes["b_continuation_ambiguous"] == 1


def test_continuation_before_any_response_is_left_unattached() -> None:
    """A b+ with no chain to extend is noted and not attached."""
    _, links, notes = run_links([("p", "1a", "s^cs"), ("r", "1b+", "s")])
    assert links["p"]["responder_chains"] == []
    assert links["p"]["status"] == "no_pair"
    assert notes["b_continuation_without_chain"] == 1


def test_recycled_pair_numbers_do_not_cross_link() -> None:
    """When numbering restarts at 1 each proposal keeps only its own response."""
    rows, links, notes = run_links(
        [
            ("p1", "1a", "s^cs"),
            ("r1", "1b", "s"),
            ("p2", "2a", "s^co"),
            ("r2", "2b", "s"),
            ("p3", "1a", "s^cs"),
            ("r3", "1b", "s"),
        ]
    )
    assert links["p1"]["responder_chains"] == [["r1"]]
    assert links["p2"]["responder_chains"] == [["r2"]]
    assert links["p3"]["responder_chains"] == [["r3"]]
    assert [links[key]["pair_group"] for key in ("p1", "p2", "p3")] == ["m1:1@0", "m1:2@2", "m1:1@4"]
    assert notes["numbering_restarts"] == 1
    assert check_links(rows, list(links.values())) == []


def test_late_response_after_a_restart_does_not_attach_to_the_old_pair() -> None:
    """A b for a number from before the restart is dropped, not added to the old proposal."""
    _, links, notes = run_links(
        [
            ("p1", "1a", "s^cs"),
            ("p2", "2a", "s^cs"),
            ("r2", "2b", "s"),
            ("p3", "1a", "s^cs"),
            ("late", "2b", "s"),
        ]
    )
    assert links["p2"]["responder_chains"] == [["r2"]]
    assert notes["b_without_group"] == 1


def test_reopened_number_without_a_restart_replaces_the_open_pair() -> None:
    """Reusing number 1 before any higher number opens a fresh pair for the new proposal."""
    _, links, _ = run_links(
        [("p1", "1a", "s^cs"), ("r1", "1b", "s"), ("p2", "1a", "s^cs"), ("r2", "1b", "s")]
    )
    assert links["p1"]["responder_chains"] == [["r1"]]
    assert links["p2"]["responder_chains"] == [["r2"]]


def test_response_without_any_open_pair_is_noted_not_attached() -> None:
    """A b with no matching a is counted in the audit notes and attached nowhere."""
    _, links, notes = run_links([("p", "1a", "s^cs"), ("r", "7b", "s")])
    assert links["p"]["responder_chains"] == []
    assert notes["b_without_group"] == 1


def test_proposal_without_adjacency_has_no_pair() -> None:
    """A proposal with no adjacency attribute is no_pair with an empty record."""
    _, links, _ = run_links([("p", None, "s^cs")])
    link = links["p"]
    assert (link["status"], link["has_response"], link["responder_chains"]) == ("no_pair", False, [])
    assert (link["pair_group"], link["adjacency_tokens"]) == (None, [])


def test_opened_pair_without_a_response_is_no_pair_but_keeps_its_group() -> None:
    """A proposal that opens a pair nobody answers is no_pair and names its group."""
    _, links, _ = run_links([("p", "3a", "s^cs"), ("x", None, "s")])
    assert links["p"]["status"] == "no_pair"
    assert links["p"]["pair_group"] == "m1:3@0"


@pytest.mark.parametrize("adjacency", ["FBH", "", "1a.FBH", "5x"])
def test_unreadable_adjacency_is_malformed_never_no_pair(adjacency: str) -> None:
    """Any unreadable adjacency piece makes the record malformed and keeps the raw tokens."""
    _, links, _ = run_links([("p", adjacency, "s^cs"), ("r", "1b", "s")])
    assert links["p"]["status"] == "malformed"
    assert links["p"]["adjacency_tokens"] == adjacency.split(".")


def test_other_speaker_continuation_without_a_pair_is_ambiguous() -> None:
    """An a-2 with no open pair creates an ambiguous group instead of a clean pair."""
    _, links, notes = run_links([("p", "1a-2", "s^cs")])
    assert links["p"]["status"] == "ambiguous"
    assert notes["a_continuation_without_group"] == 1


def test_proposal_side_continuation_shares_the_pair_with_the_first_proposal() -> None:
    """An a+ proposal joins the open pair and sees the same chains and group."""
    _, links, _ = run_links([("p1", "1a", "s^cs"), ("p2", "1a+", "s^co"), ("r", "1b", "s")])
    assert links["p1"]["responder_chains"] == links["p2"]["responder_chains"] == [["r"]]
    assert links["p1"]["pair_group"] == links["p2"]["pair_group"] == "m1:1@0"


def test_piped_response_adds_every_part_to_the_chain() -> None:
    """Tokens apply to the whole act, so all parts of a piped response join the chain."""
    _, links, _ = run_links([("p", "1a", "s^cs"), ("r", "1b", ["s^aa", "s^bk"])])
    assert links["p"]["responder_chains"] == [["r_p1", "r_p2"]]


def test_piped_proposal_links_only_its_proposal_parts() -> None:
    """Only the part tagged as a proposal gets a link record."""
    rows, links, _ = run_links([("p", "1a", ["s^cs", "s^aa"]), ("r", "1b", "s")])
    assert list(links) == ["p_p1"]
    assert links["p_p1"]["responder_chains"] == [["r"]]
    assert check_links(rows, list(links.values())) == []


def test_two_proposal_parts_of_one_act_share_one_token_set() -> None:
    """With one token for two proposal parts both records point at the same pair."""
    _, links, _ = run_links([("p", "1a", ["s^cs", "s^co"]), ("r", "1b", "s")])
    assert set(links) == {"p_p1", "p_p2"}
    assert links["p_p1"]["pair_group"] == links["p_p2"]["pair_group"]
    assert links["p_p1"]["responder_chains"] == links["p_p2"]["responder_chains"] == [["r"]]
    assert links["p_p1"]["adjacency_tokens"] == links["p_p2"]["adjacency_tokens"] == ["1a"]


def test_act_with_two_tokens_can_answer_one_pair_and_open_another() -> None:
    """A single act carrying 1b.2a answers pair 1 and opens pair 2 for its own proposal."""
    _, links, _ = run_links(
        [("p1", "1a", "s^cs"), ("x", "1b.2a", "s^cs"), ("r2", "2b", "s")]
    )
    assert links["p1"]["responder_chains"] == [["x"]]
    assert links["x"]["responder_chains"] == [["r2"]]


def test_link_records_match_the_proposal_rows_exactly() -> None:
    """The proposal ids in the links are exactly the proposal rows, in row order."""
    rows, links, _ = run_links(
        [("a", None, "s"), ("p1", "1a", "s^cs"), ("r", "1b", "s"), ("p2", None, "s^arp^cs"), ("b", None, "s^aa^cc")]
    )
    expected = [row["seg_id"] for row in rows if row["is_layer2_proposal"]]
    assert list(links) == expected == ["p1", "p2"]


def test_every_chain_id_exists_in_the_rows() -> None:
    """Every seg_id in a responder chain is the id of a real row."""
    rows, links, _ = run_links(
        [("p", "1a", "s^cs"), ("r1", "1b-1", ["s", "s^aa"]), ("r2", "1b-2", "s"), ("c", "1b-1+", "s")]
    )
    known = {row["seg_id"] for row in rows}
    members = {seg_id for link in links.values() for chain in link["responder_chains"] for seg_id in chain}
    assert members and members <= known


def test_every_scenario_passes_the_link_contract() -> None:
    """A busy meeting produces links that satisfy every contract check."""
    rows, links, _ = run_links(
        [
            ("p1", "1a", "s^cs"),
            ("r1", "1b-1", "s"),
            ("r2", "1b-2", "s"),
            ("c", "1b-1+", "s"),
            ("p2", "2a", "s^co"),
            ("m", "2b", ["s", "s^aa"]),
            ("p3", None, "s^cs"),
            ("p4", "FBH", "s^cs"),
        ]
    )
    assert check_links(rows, list(links.values())) == []
