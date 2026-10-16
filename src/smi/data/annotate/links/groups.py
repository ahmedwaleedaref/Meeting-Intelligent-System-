"""Build adjacency-pair groups and responder chains from one meeting's ordered rows."""

from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby
from typing import Sequence

from smi.data.annotate.links.tokens import Token, parse_adjacency
from smi.data.annotate.records import ParsedRow

GroupKey = tuple[int, int]  # (pair number, underscore count)


@dataclass(eq=False)
class PairGroup:
    """One adjacency pair. `chains` maps a responder's speaker number (0 = none) to seg_ids."""

    group_id: str
    chains: dict[int, list[str]] = field(default_factory=dict)
    ambiguous: bool = False


@dataclass
class Grouping:
    """Groups opened or extended by each source act (by `source_da_id`), plus audit counts."""

    by_element: dict[str, list[PairGroup]] = field(default_factory=dict)
    notes: Counter[str] = field(default_factory=Counter)


@dataclass
class _State:
    grouping: Grouping = field(default_factory=Grouping)
    open_groups: dict[GroupKey, PairGroup] = field(default_factory=dict)
    highest: int = 0


def build_groups(rows: Sequence[ParsedRow]) -> Grouping:
    """Walk one meeting's rows in position order and build its pair groups.

    The parts of a source act are consecutive rows that repeat its adjacency string, so
    each act is read once. Tokens apply to the whole act (manual 4.3), never to one part.
    """
    state = _State()
    for _, parts in groupby(rows, key=lambda row: row["source_da_id"]):
        element = list(parts)
        tokens, _unparsed = parse_adjacency(element[0]["adjacency_raw"])
        for token in tokens:
            if token.role == "a":
                _apply_a(state, token, element)
            else:
                _apply_b(state, token, element)
    return state.grouping


def _apply_a(state: _State, token: Token, element: list[ParsedRow]) -> None:
    """An `a` opens a pair; `a+` or another speaker's `a-n` (n > 1) extends the open one."""
    key = (token.number, token.underscores)
    opens = token.plus == 0 and token.speaker in (None, 1)
    # Numbering restarts at 1 in every labelled chunk (manual 4.2; paper: 10-minute chunks).
    if opens and token.underscores == 0 and token.number == 1 and state.highest > 1:
        state.open_groups.clear()
        state.highest = 0
        state.grouping.notes["numbering_restarts"] += 1
    if token.underscores == 0:
        state.highest = max(state.highest, token.number)

    group = None if opens else state.open_groups.get(key)
    if group is None:
        first = element[0]
        suffix = "_" * token.underscores
        group = PairGroup(
            group_id=f"{first['meeting']}:{token.number}{suffix}@{first['position']}",
            ambiguous=not opens,
        )
        state.open_groups[key] = group
        if not opens:
            state.grouping.notes["a_continuation_without_group"] += 1

    groups = state.grouping.by_element.setdefault(element[0]["source_da_id"], [])
    if all(known is not group for known in groups):
        groups.append(group)


def _apply_b(state: _State, token: Token, element: list[ParsedRow]) -> None:
    """A `b` adds this act's parts to its speaker's response chain in the open pair."""
    notes = state.grouping.notes
    group = state.open_groups.get((token.number, token.underscores))
    if group is None:
        notes["b_without_group"] += 1
        return
    speaker = _chain_speaker(group, token)
    if speaker is None and group.chains:
        notes["b_continuation_ambiguous"] += 1
        group.ambiguous = True
        return
    if speaker is None:
        notes["b_continuation_without_chain"] += 1
        return
    chain = group.chains.setdefault(speaker, [])
    chain.extend(row["seg_id"] for row in element if row["seg_id"] not in chain)


def _chain_speaker(group: PairGroup, token: Token) -> int | None:
    """Chain a `b` extends: its own `-n`, else chain 0, else the only chain (bare `b+`)."""
    if token.speaker is not None or token.plus == 0 or 0 in group.chains:
        return token.speaker or 0
    return next(iter(group.chains)) if len(group.chains) == 1 else None
