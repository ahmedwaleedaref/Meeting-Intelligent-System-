"""Turn each Layer-2 proposal row into one link record: responder chains plus a status."""

from typing import Final, Sequence

from smi.data.annotate.links.groups import Grouping, PairGroup
from smi.data.annotate.links.tokens import parse_adjacency
from smi.data.annotate.records import AnnotatedRow, LinkRecord

OK: Final = "ok"
NO_PAIR: Final = "no_pair"
MALFORMED: Final = "malformed"
AMBIGUOUS: Final = "ambiguous"


def build_links(rows: Sequence[AnnotatedRow], grouping: Grouping) -> list[LinkRecord]:
    position_of = {row["seg_id"]: row["position"] for row in rows}
    return [
        _link_record(row, grouping.by_element.get(row["source_da_id"], []), position_of)
        for row in rows
        if row["is_layer2_proposal"]
    ]


def _link_record(
    row: AnnotatedRow, groups: list[PairGroup], position_of: dict[str, int]
) -> LinkRecord:
    adjacency_raw = row["adjacency_raw"]
    _tokens, unparsed = parse_adjacency(adjacency_raw)
    chains = sorted(
        (list(chain) for group in groups for chain in group.chains.values()),
        key=lambda chain: position_of[chain[0]],
    )
    return {
        "proposal_id": row["seg_id"],
        "responder_chains": chains,
        "has_response": bool(chains),
        "status": _status(unparsed, groups),
        "adjacency_tokens": adjacency_raw.split(".") if adjacency_raw is not None else [],
        "pair_group": groups[0].group_id if groups else None,
    }


def _status(unparsed: list[str], groups: list[PairGroup]) -> str:
    if unparsed:
        return MALFORMED
    if any(group.ambiguous for group in groups):
        return AMBIGUOUS
    return OK if any(group.chains for group in groups) else NO_PAIR
