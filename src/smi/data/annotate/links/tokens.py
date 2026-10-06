"""Read adjacency-pair tokens into typed values."""

import re
from typing import Final, NamedTuple

_TOKEN: Final = re.compile(r"(\d+)(_*)([ab])(?:-(\d+))?(\+*)")


class Token(NamedTuple):
    raw: str
    number: int
    underscores: int
    role: str
    speaker: int | None
    plus: int


def parse_token(piece: str) -> Token | None:
    match = _TOKEN.fullmatch(piece)
    if match is None:
        return None
    number, underscores, role, speaker, plus = match.groups()
    return Token(
        raw=piece,
        number=int(number),
        underscores=len(underscores),
        role=role,
        speaker=int(speaker) if speaker else None,
        plus=len(plus),
    )


def parse_adjacency(adjacency_raw: str | None) -> tuple[list[Token], list[str]]:
    pieces = adjacency_raw.split(".") if adjacency_raw is not None else []
    parsed = [(piece, parse_token(piece)) for piece in pieces]
    tokens = [token for _, token in parsed if token is not None]
    unparsed = [piece for piece, token in parsed if token is None]
    return tokens, unparsed
