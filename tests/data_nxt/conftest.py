"""Fixtures that parse the hand-written NXT corpus once per test."""

from pathlib import Path
from typing import Any

import pytest

from nxt_fixtures import build_standard_corpus
from smi.data.nxt.parser import parse_corpus


@pytest.fixture
def standard_corpus(tmp_path: Path) -> Path:
    return build_standard_corpus(tmp_path / "raw")


@pytest.fixture
def standard_parse(standard_corpus: Path) -> tuple[list[dict[str, Any]], Any, list[dict[str, str]]]:
    return parse_corpus(standard_corpus)


@pytest.fixture
def standard_rows(standard_parse: tuple[list[dict[str, Any]], Any, Any]) -> dict[str, dict[str, Any]]:
    rows = standard_parse[0]
    return {row["seg_id"]: row for row in rows}
