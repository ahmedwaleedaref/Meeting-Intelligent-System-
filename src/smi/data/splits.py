"""Build reproducible meeting-level split and cross-validation assignments."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from xml.etree import ElementTree


SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FOLD_COUNT = 5
DEFAULT_SPLIT_COUNTS = {"train": 51, "val": 12, "test": 12}
DEFAULT_SPLIT_SEED = "icsi-mrda-split-v1"


def load_meeting_ids(metadata_path: str | Path) -> list[str]:
    """Read only the observation names that define the NXT meeting universe."""
    root = ElementTree.parse(metadata_path).getroot()
    observations = None
    for element in root.iter():
        tag_name = element.tag.rsplit("}", 1)[-1]
        if tag_name == "observations":
            observations = element
            break

    if observations is None:
        raise ValueError("metadata XML has no <observations> element")

    meeting_ids = []
    for element in observations:
        if element.tag.rsplit("}", 1)[-1] != "observation":
            continue
        meeting_id = element.attrib.get("name", "").strip()
        if not meeting_id:
            raise ValueError("an <observation> is missing its name")
        meeting_ids.append(meeting_id)

    if not meeting_ids:
        raise ValueError("metadata XML has no <observation> entries")
    if len(meeting_ids) != len(set(meeting_ids)):
        raise ValueError("metadata XML contains duplicate observation names")
    return meeting_ids


def assign_grouped_folds(meeting_ids: Iterable[str], k: int = DEFAULT_FOLD_COUNT) -> dict[str, int]:
    """Assign each meeting to one balanced fold in a stable, input-order-free way.

    The meetings are sorted lexicographically and distributed round-robin. Because
    a meeting is the assignment unit, rows from a meeting cannot cross folds.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer of at least 2")

    meetings = list(meeting_ids)
    for meeting in meetings:
        if not isinstance(meeting, str) or not meeting:
            raise ValueError("meeting IDs must be non-empty strings")

    if len(meetings) != len(set(meetings)):
        raise ValueError("meeting IDs must be unique")

    assignments = {}
    for index, meeting in enumerate(sorted(meetings)):
        assignments[meeting] = index % k
    return assignments


def assign_stratified_splits(
    meeting_ids: Iterable[str],
    split_counts: Mapping[str, int] = DEFAULT_SPLIT_COUNTS,
    *,
    seed: str = DEFAULT_SPLIT_SEED,
) -> dict[str, list[str]]:
    """Assign whole meetings reproducibly, balancing each meeting series.

    Meetings within a series are ordered by a seeded SHA-256 value. Series
    counts are allocated as close as possible to the requested split ratios,
    while meeting the exact train, validation, and test totals.
    """
    meetings = list(meeting_ids)
    if not meetings or len(meetings) != len(set(meetings)):
        raise ValueError("meeting_ids must be non-empty and unique")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a non-empty string")
    if set(split_counts) != set(SPLIT_NAMES):
        raise ValueError(f"split_counts must contain exactly {', '.join(SPLIT_NAMES)}")
    for split in SPLIT_NAMES:
        count = split_counts[split]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{split} count must be a non-negative integer")
    if sum(split_counts.values()) != len(meetings):
        raise ValueError("split counts must add up to the number of meetings")

    meetings_by_series: dict[str, list[str]] = {}
    for meeting in meetings:
        if not isinstance(meeting, str) or not meeting:
            raise ValueError("meeting IDs must be non-empty strings")
        match = re.match(r"[A-Za-z]+", meeting)
        if match is None:
            raise ValueError(f"cannot determine meeting series for {meeting!r}")
        series = match.group(0)
        meetings_by_series.setdefault(series, []).append(meeting)

    total_meetings = len(meetings)
    allocation = {}
    target_totals = {split: split_counts[split] for split in SPLIT_NAMES}
    assigned_totals = {split: 0 for split in SPLIT_NAMES}
    quotas = {}

    for series in sorted(meetings_by_series):
        series_size = len(meetings_by_series[series])
        quotas[series] = {}
        allocation[series] = {}
        for split in SPLIT_NAMES:
            quota = series_size * split_counts[split] / total_meetings
            count = int(quota)
            quotas[series][split] = quota
            allocation[series][split] = count
            assigned_totals[split] += count

    remaining_totals = {}
    for split in SPLIT_NAMES:
        remaining_totals[split] = target_totals[split] - assigned_totals[split]

    while sum(remaining_totals.values()) > 0:
        best_choice = None
        best_priority = None

        for series in sorted(meetings_by_series):
            allocated_in_series = sum(allocation[series].values())
            if allocated_in_series == len(meetings_by_series[series]):
                continue

            for split in SPLIT_NAMES:
                if remaining_totals[split] == 0:
                    continue
                priority = quotas[series][split] - allocation[series][split]
                choice = (series, split)
                if best_priority is None or priority > best_priority:
                    best_choice = choice
                    best_priority = priority

        if best_choice is None:
            raise ValueError("could not satisfy requested split counts")

        series, split = best_choice
        allocation[series][split] += 1
        remaining_totals[split] -= 1

    output = {split: [] for split in SPLIT_NAMES}
    for series in sorted(meetings_by_series):
        series_meetings = sorted(
            meetings_by_series[series],
            key=lambda meeting: hashlib.sha256(f"{seed}:{meeting}".encode("utf-8")).hexdigest(),
        )
        start = 0
        for split in SPLIT_NAMES:
            count = allocation[series][split]
            output[split].extend(series_meetings[start : start + count])
            start += count

    for split in SPLIT_NAMES:
        output[split].sort()
    return output


def build_split_provenance(
    meeting_universe: Iterable[str],
    split_meetings: Mapping[str, Iterable[str]],
    *,
    split_source: str,
    fold_count: int = DEFAULT_FOLD_COUNT,
    reasons: Mapping[str, str] | None = None,
    universe_source: str = "ICSI-metadata.xml <observation> names",
) -> dict[str, object]:
    """Create the versioned provenance object after validating complete coverage.

    ``split_meetings`` must be the authoritative train/val/test lists supplied
    by the project lead. No split membership is inferred by this function.
    """
    if not split_source.strip():
        raise ValueError("split_source must identify the authoritative lists")
    if not universe_source.strip():
        raise ValueError("universe_source must identify the meeting universe")
    if set(split_meetings) != set(SPLIT_NAMES):
        raise ValueError(f"split_meetings must contain exactly {', '.join(SPLIT_NAMES)}")

    universe = list(meeting_universe)
    for meeting in universe:
        if not isinstance(meeting, str) or not meeting:
            raise ValueError("meeting IDs must be non-empty strings")

    if len(universe) != len(set(universe)):
        raise ValueError("meeting_universe contains duplicate IDs")
    universe_set = set(universe)

    normalized: dict[str, list[str]] = {}
    assignment: dict[str, str] = {}
    for split in SPLIT_NAMES:
        values = list(split_meetings[split])
        for meeting in values:
            if not isinstance(meeting, str) or not meeting:
                raise ValueError(f"{split} meeting IDs must be non-empty strings")

        if len(values) != len(set(values)):
            raise ValueError(f"{split} contains duplicate meeting IDs")
        for meeting in values:
            if meeting not in universe_set:
                raise ValueError(f"{meeting!r} in {split} is absent from the meeting universe")
            if meeting in assignment:
                raise ValueError(f"{meeting!r} appears in more than one split")
            assignment[meeting] = split
        normalized[split] = values

    missing = sorted(universe_set - set(assignment))
    if missing:
        raise ValueError(f"meetings missing from split lists: {', '.join(missing)}")

    practice_meetings = normalized["train"] + normalized["val"]
    fold_assignments = assign_grouped_folds(practice_meetings, fold_count)
    reason_map = reasons or {}
    records = []
    for meeting in sorted(universe):
        split = assignment[meeting]
        reason = reason_map.get(meeting, f"Table 9 {split} list")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"reason for {meeting!r} must be a non-empty string")
        records.append(
            {
                "meeting_id": meeting,
                "split": split,
                "reason": reason,
                "fold": fold_assignments.get(meeting),
            }
        )

    return {
        "version": "v1",
        "sources": {
            "split_lists": split_source,
            "meeting_universe": universe_source,
        },
        "folds": {
            "k": fold_count,
            "grouping": "meeting",
            "method": (
                "Sort train and val meeting IDs lexicographically, then assign "
                "round-robin to folds 0 through k-1."
            ),
            "seed": None,
        },
        "meetings": records,
    }


def attach_split_info(
    rows: Iterable[Mapping[str, object]],
    split_provenance: Mapping[str, object],
) -> list[dict[str, object]]:
    """Copy C2 rows and append their meeting-level ``split`` and ``fold`` fields."""
    fold_config = split_provenance.get("folds")
    if not isinstance(fold_config, Mapping):
        raise ValueError("split provenance is missing its folds configuration")
    fold_count = fold_config.get("k")
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError("split provenance has an invalid fold count")

    meeting_records = split_provenance.get("meetings")
    if not isinstance(meeting_records, list):
        raise ValueError("split provenance is missing its meeting records")

    assignments: dict[str, tuple[str, int | None]] = {}
    for record in meeting_records:
        if not isinstance(record, Mapping):
            raise ValueError("each meeting assignment must be an object")
        meeting = record.get("meeting_id")
        split = record.get("split")
        fold = record.get("fold")
        if not isinstance(meeting, str) or not meeting:
            raise ValueError("meeting assignment has an invalid meeting_id")
        if meeting in assignments:
            raise ValueError(f"duplicate split assignment for meeting {meeting!r}")
        if split not in SPLIT_NAMES:
            raise ValueError(f"invalid split for meeting {meeting!r}: {split!r}")
        if split == "test":
            if fold is not None:
                raise ValueError(f"test meeting {meeting!r} must have fold=null")
        elif isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < fold_count:
            raise ValueError(f"train/val meeting {meeting!r} has an invalid fold")
        assignments[meeting] = (split, fold)

    output = []
    for row in rows:
        if "split" in row or "fold" in row:
            raise ValueError("input C2 row already contains split or fold fields")
        meeting = row.get("meeting")
        if not isinstance(meeting, str) or not meeting:
            raise ValueError("input C2 row is missing a valid meeting field")
        if meeting not in assignments:
            raise ValueError(f"no split assignment found for meeting {meeting!r}")
        split, fold = assignments[meeting]
        updated_row = dict(row)
        updated_row["split"] = split
        updated_row["fold"] = fold
        output.append(updated_row)
    return output


def write_split_provenance(provenance: Mapping[str, object], path: str | Path) -> None:
    """Write stable JSON without build-time metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
