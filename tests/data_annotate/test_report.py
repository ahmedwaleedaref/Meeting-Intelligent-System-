"""Task 2: the generated annotation report."""

import re
from collections import Counter

from annotate_fixtures import meeting_rows, run_links
from smi.data.annotate.records import LinkRecord
from smi.data.annotate.report import SPEC_MULTI_TARGET, WINDOW, build_report

SPECS = [
    ("p", "1a", "s^cs"),
    ("r", "1b", "s^aa"),
    ("q", None, "s^arp^cs"),
    ("m", "FBH", "s^co"),
]


def report_for(specs: list[tuple[str, str | None, str]]) -> str:
    rows, links, notes = run_links(specs)
    return build_report(rows, list(links.values()), notes)


def test_report_counts_rows_and_labels() -> None:
    """The report states the row total and a count for every label."""
    text = report_for(SPECS)
    assert "- Annotated rows: 4" in text
    for line in ("| cs | 2 |", "| co | 1 |", "| aa | 1 |", "| bk | 0 |", "| ar | 0 |", "| cc | 0 |", "| other | 0 |"):
        assert line in text


def test_report_counts_flags_and_the_bk_variant_shift() -> None:
    """The flag table and the b/bk shift line are filled from the rows."""
    text = report_for(SPECS)
    assert "| is_layer2_proposal | 3 |" in text
    assert "| is_nonspeech | 0 |" in text
    assert "- Rows whose label changes under the b/bk variant: 0" in text
    assert "- Rows carrying an error-class quality flag: 0" in text


def test_report_compares_the_multi_target_count_with_the_spec() -> None:
    """A multi-target count other than 18 is reported as differing from the spec."""
    text = report_for(SPECS)
    assert f"- 1 rows; the spec expects {SPEC_MULTI_TARGET} (differs from the spec)." in text
    assert "| cs, ar | 1 |" in text


def test_report_says_matches_when_the_multi_target_count_is_18() -> None:
    """Exactly 18 multi-target rows are reported as matching the spec."""
    text = report_for([(f"q{index}", None, "s^arp^cs") for index in range(SPEC_MULTI_TARGET)])
    assert f"- {SPEC_MULTI_TARGET} rows; the spec expects {SPEC_MULTI_TARGET} (matches the spec)." in text


def test_report_counts_link_statuses_and_chains() -> None:
    """The report counts links by status and ok records by number of chains."""
    text = report_for(SPECS)
    assert "- Link records (one per Layer-2 proposal row): 3" in text
    for line in ("| ok | 1 |", "| no_pair | 1 |", "| malformed | 1 |", "| ambiguous | 0 |"):
        assert line in text
    assert "| 1 | 1 |" in text


def test_report_lists_the_pair_group_audit_notes() -> None:
    """Audit notes are listed by name with their counts."""
    rows, links, _ = run_links(SPECS)
    text = build_report(rows, list(links.values()), Counter({"b_without_group": 3, "numbering_restarts": 2}))
    assert "| b_without_group | 3 |" in text and "| numbering_restarts | 2 |" in text


def test_report_states_the_share_of_proposals_reachable_in_the_window() -> None:
    """A response one row away is reachable within the window and counted as 100%."""
    text = report_for([("p", "1a", "s^cs"), ("r", "1b", "s")])
    assert f"looks at the next {WINDOW} rows can reach 1 of 1 (100.0%)" in text
    assert "| 1-5 | 1 |" in text


def test_report_buckets_response_distances() -> None:
    """Responses further than the window land in the matching distance bucket."""
    fillers = lambda count: [(f"f{index}", None, "s") for index in range(count)]
    text = report_for([("p", "1a", "s^cs"), *fillers(6), ("r", "1b", "s")])
    assert "| 6-10 | 1 |" in text and "| 1-5 | 0 |" in text
    text = report_for([("p", "1a", "s^cs"), *fillers(12), ("r", "1b", "s")])
    assert "| 11-50 | 1 |" in text
    text = report_for([("p", "1a", "s^cs"), *fillers(60), ("r", "1b", "s")])
    assert "| more than 50 | 1 |" in text


def test_report_handles_the_case_without_ok_links() -> None:
    """With no ok record the reachable share is shown as n/a."""
    text = report_for([("p", None, "s^cs")])
    assert "(n/a)" in text


def test_report_counts_responses_that_are_not_after_the_proposal() -> None:
    """A link whose response is not after the proposal goes to the no-response bucket."""
    rows = meeting_rows([("r", None, "s"), ("p", None, "s^cs")])
    link = LinkRecord(
        proposal_id="p",
        responder_chains=[["r"]],
        has_response=True,
        status="ok",
        adjacency_tokens=[],
        pair_group="m1:1@1",
    )
    text = build_report(rows, [link], Counter())
    assert "| no response after the proposal | 1 |" in text


def test_report_is_reproducible_and_has_no_timestamp() -> None:
    """The same inputs give the same text, with no date in it."""
    text = report_for(SPECS)
    assert text == report_for(SPECS)
    assert re.search(r"\d{4}-\d{2}-\d{2}", text) is None
