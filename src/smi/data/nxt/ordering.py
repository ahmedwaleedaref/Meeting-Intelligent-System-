"""The fixed C1 row ordering defined by D12."""

from collections import defaultdict
from typing import Any


def assign_positions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rows by the source-derived D12 key and assign meeting-local ranks."""
    rows.sort(
        key=lambda row: (
            row["meeting"],
            row["src_start"],
            row["src_end"],
            row["agent"],
            row["element_ordinal"],
            row["part_index"],
        )
    )
    next_position: dict[str, int] = defaultdict(int)
    for row in rows:
        meeting = row["meeting"]
        row["position"] = next_position[meeting]
        next_position[meeting] += 1
    return rows
