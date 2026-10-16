"""Task 2: adjacency token parsing."""

import pytest

from smi.data.annotate.links.tokens import Token, parse_adjacency, parse_token


@pytest.mark.parametrize(
    ("piece", "expected"),
    [
        ("5a", Token("5a", 5, 0, "a", None, 0)),
        ("5b", Token("5b", 5, 0, "b", None, 0)),
        ("5b+", Token("5b+", 5, 0, "b", None, 1)),
        ("14b++", Token("14b++", 14, 0, "b", None, 2)),
        ("5b-1", Token("5b-1", 5, 0, "b", 1, 0)),
        ("5b-2+", Token("5b-2+", 5, 0, "b", 2, 1)),
        ("8_b", Token("8_b", 8, 1, "b", None, 0)),
    ],
)
def test_valid_tokens_are_parsed_into_typed_fields(piece: str, expected: Token) -> None:
    """Number, underscores, role, speaker and plus count are read from a token."""
    assert parse_token(piece) == expected


@pytest.mark.parametrize("piece", ["FBH", "", "5c", "a5", "5b-", "b", "5 b", "5bb"])
def test_unrecognised_tokens_are_rejected(piece: str) -> None:
    """Tokens outside the number-role pattern parse to None."""
    assert parse_token(piece) is None


def test_adjacency_is_split_on_dots() -> None:
    """A dotted adjacency string yields one token per piece in order."""
    tokens, unparsed = parse_adjacency("14b++.15a")
    assert [token.raw for token in tokens] == ["14b++", "15a"]
    assert unparsed == []


def test_unparsed_pieces_are_reported_next_to_parsed_ones() -> None:
    """A bad piece is returned in the unparsed list while good pieces still parse."""
    tokens, unparsed = parse_adjacency("5b.FBH")
    assert [token.raw for token in tokens] == ["5b"]
    assert unparsed == ["FBH"]


def test_missing_adjacency_has_no_tokens_and_nothing_unparsed() -> None:
    """None means no adjacency at all, which is not a malformed value."""
    assert parse_adjacency(None) == ([], [])


def test_empty_adjacency_string_is_unparsed() -> None:
    """An empty adjacency string is reported as an unparsed piece."""
    assert parse_adjacency("") == ([], [""])
