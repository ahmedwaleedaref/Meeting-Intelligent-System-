"""Validate and load the frozen C3 release and expose its Layer-2 gold view."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smi.labels import LABELS


RELEASE_FILES = {
    "rows.jsonl",
    "links.jsonl",
    "splits.json",
    "metadata.json",
    "checksums.sha256",
}
_CONTENT_FILES = ("links.jsonl", "metadata.json", "rows.jsonl", "splits.json")
_SPLITS = {"train", "val", "test"}
_LINK_STATUSES = {"ok", "no_pair", "malformed", "ambiguous"}
_ERROR_QUALITY_FLAGS = {
    "ref_unresolved",
    "ref_malformed",
    "time_invalid",
    "type_unavailable",
}
_ROW_FIELDS = {
    "seg_id",
    "meeting",
    "agent",
    "channel",
    "speaker_id",
    "src_start",
    "src_end",
    "start",
    "end",
    "timing_method",
    "position",
    "source_da_id",
    "element_ordinal",
    "part_index",
    "n_parts",
    "raw_type",
    "original_type",
    "part_type",
    "adjacency_raw",
    "comment",
    "text",
    "words",
    "quality_flags",
    "label",
    "label_bk_merged",
    "target_tags",
    "is_multi_target",
    "is_layer2_proposal",
    "general_tags",
    "disruption_markers",
    "is_nonspeech",
    "is_empty",
    "split",
    "fold",
}
_LINK_FIELDS = {
    "proposal_id",
    "responder_chains",
    "has_response",
    "status",
    "adjacency_tokens",
    "pair_group",
}
_WORD_FIELDS = {"text", "kind", "start", "end"}


@dataclass(frozen=True)
class ReleaseData:
    """Validated release contents loaded from a ``data/v1`` directory."""

    rows: tuple[dict[str, Any], ...]
    links: tuple[dict[str, Any], ...]
    splits: dict[str, Any]
    metadata: dict[str, Any]

    def layer2_gold(self) -> tuple[dict[str, Any], ...]:
        """Return eligible, resolvable proposal records with their acceptable sets."""
        rows_by_id = {}
        for row in self.rows:
            rows_by_id[row["seg_id"]] = row

        result = []
        for record in self.links:
            if record["status"] not in {"ok", "no_pair"}:
                continue
            result.append(
                {
                    "proposal_id": record["proposal_id"],
                    "acceptable_responses": acceptable_responses(record),
                    "proposal": rows_by_id[record["proposal_id"]],
                    "link": record,
                }
            )
        return tuple(result)


def acceptable_responses(
    record: Mapping[str, Any],
    *,
    responders: str = "all",
    segments: str = "first",
) -> set[str]:
    """Derive the benchmark acceptable-response set from stored responder chains.

    ``responders`` is ``all`` or ``first``; ``segments`` is ``first`` or ``any``.
    Malformed and ambiguous records are excluded rather than interpreted as none.
    """
    if responders not in ("all", "first"):
        raise ValueError("responders must be 'all' or 'first'")
    if segments not in ("first", "any"):
        raise ValueError("segments must be 'first' or 'any'")
    status = record.get("status")
    if status not in ("ok", "no_pair"):
        raise ValueError(f"status {status!r} is not eligible for Layer-2 gold")

    chains = record.get("responder_chains")
    if not isinstance(chains, list):
        raise ValueError("responder_chains must be a list of lists of non-empty seg_id strings")
    for chain in chains:
        if not isinstance(chain, list):
            raise ValueError("responder_chains must be a list of lists of non-empty seg_id strings")
        for seg_id in chain:
            if not isinstance(seg_id, str) or not seg_id:
                raise ValueError(
                    "responder_chains must be a list of lists of non-empty seg_id strings"
                )

    if (status == "no_pair") != (len(chains) == 0):
        raise ValueError("status and responder_chains disagree")

    if responders == "first":
        selected_chains = chains[:1]
    else:
        selected_chains = chains

    accepted = set()
    for chain in selected_chains:
        if segments == "first":
            segments_in_chain = chain[:1]
        else:
            segments_in_chain = chain
        for seg_id in segments_in_chain:
            accepted.add(seg_id)
    return accepted


def validate_release_content(
    rows: Sequence[Mapping[str, Any]],
    links: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Any],
) -> None:
    """Raise ``ValueError`` if rows, links, and split assignments disagree."""
    fold_count = splits.get("k")
    if type(fold_count) is not int or fold_count < 2:
        raise ValueError("splits.k must be an integer of at least 2")
    if splits.get("grouping") != "meeting":
        raise ValueError("splits.grouping must be 'meeting'")
    meeting_assignments = splits.get("meetings")
    if not isinstance(meeting_assignments, Mapping):
        raise ValueError("splits.meetings must map meeting IDs to split/fold assignments")
    source_hash = splits.get("source_sha256")
    if not _is_sha256(source_hash):
        raise ValueError("splits.source_sha256 must be a lowercase SHA-256 fingerprint")

    rows_by_id: dict[str, Mapping[str, Any]] = {}
    meeting_seen: dict[str, tuple[str, int | None]] = {}
    positions: set[tuple[str, int]] = set()
    positions_by_meeting: dict[str, list[int]] = {}
    ordered_keys: list[tuple[str, int]] = []
    for row in rows:
        missing = _ROW_FIELDS - set(row)
        if missing:
            raise ValueError(f"row is missing mandatory fields: {', '.join(sorted(missing))}")
        seg_id = row["seg_id"]
        meeting = row["meeting"]
        if not isinstance(seg_id, str) or not seg_id:
            raise ValueError("row seg_id must be a non-empty string")
        if seg_id in rows_by_id:
            raise ValueError(f"duplicate seg_id: {seg_id!r}")
        if not isinstance(meeting, str) or not meeting:
            raise ValueError(f"row {seg_id!r} has an invalid meeting")
        if type(row["position"]) is not int or row["position"] < 0:
            raise ValueError(f"row {seg_id!r} has an invalid position")
        if (meeting, row["position"]) in positions:
            raise ValueError(f"duplicate position in meeting {meeting!r}")
        positions.add((meeting, row["position"]))
        ordered_keys.append((meeting, row["position"]))
        if meeting not in positions_by_meeting:
            positions_by_meeting[meeting] = []
        positions_by_meeting[meeting].append(row["position"])

        text_fields = (
            "agent",
            "channel",
            "speaker_id",
            "timing_method",
            "source_da_id",
            "original_type",
            "part_type",
            "text",
        )
        for field in text_fields:
            if not isinstance(row[field], str):
                raise ValueError(f"row {seg_id!r} has invalid {field}")

        for field in ("adjacency_raw", "comment"):
            if row[field] is not None and not isinstance(row[field], str):
                raise ValueError(f"row {seg_id!r} has invalid {field}")

        for field in ("src_start", "src_end", "start", "end"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"row {seg_id!r} has invalid {field}")
            if not math.isfinite(value):
                raise ValueError(f"row {seg_id!r} has invalid {field}")

        if not (row["src_start"] <= row["start"] <= row["end"] <= row["src_end"]):
            raise ValueError(f"row {seg_id!r} has inconsistent source and dialogue-act timings")

        for field in ("element_ordinal", "part_index", "n_parts"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"row {seg_id!r} has invalid {field}")

        if row["n_parts"] < 1 or row["part_index"] >= row["n_parts"]:
            raise ValueError(f"row {seg_id!r} has inconsistent part_index and n_parts")

        words = row["words"]
        if not isinstance(words, list):
            raise ValueError(f"row {seg_id!r} has invalid words")
        for word in words:
            if not isinstance(word, Mapping):
                raise ValueError(f"row {seg_id!r} has invalid words")
            if not _WORD_FIELDS.issubset(word):
                raise ValueError(f"row {seg_id!r} has invalid words")
            if not isinstance(word["text"], str) or not isinstance(word["kind"], str):
                raise ValueError(f"row {seg_id!r} has invalid words")

        if row["label"] not in LABELS or row["label_bk_merged"] not in LABELS:
            raise ValueError(f"row {seg_id!r} has a label outside LABELS")
        target_tags = row["target_tags"]
        if not isinstance(target_tags, list):
            raise ValueError(f"row {seg_id!r} has invalid target_tags")
        for tag in target_tags:
            if not isinstance(tag, str) or tag not in LABELS[:-1]:
                raise ValueError(f"row {seg_id!r} has invalid target_tags")
        if len(target_tags) != len(set(target_tags)):
            raise ValueError(f"row {seg_id!r} has duplicate target_tags")
        canonical_tags = []
        for label in LABELS[:-1]:
            if label in target_tags:
                canonical_tags.append(label)
        if target_tags != canonical_tags:
            raise ValueError(f"row {seg_id!r} target_tags are not in canonical LABELS order")
        if type(row["is_layer2_proposal"]) is not bool:
            raise ValueError(f"row {seg_id!r} has invalid is_layer2_proposal")
        should_be_proposal = "cs" in target_tags or "co" in target_tags
        if row["is_layer2_proposal"] != should_be_proposal:
            raise ValueError(f"row {seg_id!r} has is_layer2_proposal inconsistent with target_tags")
        for name in ("is_multi_target", "is_nonspeech", "is_empty"):
            if type(row[name]) is not bool:
                raise ValueError(f"row {seg_id!r} has invalid {name}")
        if row["is_multi_target"] != (len(target_tags) >= 2):
            raise ValueError(f"row {seg_id!r} has is_multi_target inconsistent with target_tags")
        quality_flags = row["quality_flags"]
        if not isinstance(quality_flags, list):
            raise ValueError(f"row {seg_id!r} has invalid quality_flags")
        for flag in quality_flags:
            if not isinstance(flag, str):
                raise ValueError(f"row {seg_id!r} has invalid quality_flags")
        if row["raw_type"] is None:
            if "type_fallback_original" not in quality_flags:
                raise ValueError(f"row {seg_id!r} has null raw_type without a fallback flag")
        elif not isinstance(row["raw_type"], str):
            raise ValueError(f"row {seg_id!r} has invalid raw_type")
        error_flags = set()
        for flag in quality_flags:
            if flag in _ERROR_QUALITY_FLAGS:
                error_flags.add(flag)
        if error_flags:
            flags_text = ", ".join(sorted(error_flags))
            raise ValueError(f"row {seg_id!r} has error-class quality flags: {flags_text}")
        if row["is_empty"] != (row["text"] == ""):
            raise ValueError(f"row {seg_id!r} has is_empty inconsistent with text")
        for field in ("general_tags", "disruption_markers"):
            values = row[field]
            if not isinstance(values, list):
                raise ValueError(f"row {seg_id!r} has invalid {field}")
            for value in values:
                if not isinstance(value, str):
                    raise ValueError(f"row {seg_id!r} has invalid {field}")

        split = row["split"]
        fold = row["fold"]
        if not isinstance(split, str) or split not in _SPLITS:
            raise ValueError(f"row {seg_id!r} has invalid split")
        if split == "test":
            if fold is not None:
                raise ValueError(f"test row {seg_id!r} must have fold=null")
        elif type(fold) is not int or not 0 <= fold < fold_count:
            raise ValueError(f"train/val row {seg_id!r} is missing a valid fold")
        assignment = (split, fold)
        if meeting in meeting_seen and meeting_seen[meeting] != assignment:
            raise ValueError(f"meeting {meeting!r} appears in multiple splits or folds")
        meeting_seen[meeting] = assignment
        declared = meeting_assignments.get(meeting)
        if not isinstance(declared, Mapping):
            raise ValueError(f"row {seg_id!r} disagrees with splits.json")
        declared_assignment = (declared.get("split"), declared.get("fold"))
        if declared_assignment != assignment:
            raise ValueError(f"row {seg_id!r} disagrees with splits.json")
        rows_by_id[seg_id] = row

    declared_meetings = set(meeting_assignments)
    if declared_meetings != set(meeting_seen):
        missing_rows = sorted(declared_meetings - set(meeting_seen))
        extra_rows = sorted(set(meeting_seen) - declared_meetings)
        details = []
        if missing_rows:
            details.append(f"meetings with no rows: {', '.join(missing_rows)}")
        if extra_rows:
            details.append(f"rows with undeclared meetings: {', '.join(extra_rows)}")
        raise ValueError("split meeting coverage mismatch (" + "; ".join(details) + ")")
    if ordered_keys != sorted(ordered_keys):
        raise ValueError("rows.jsonl must be sorted by (meeting, position)")
    for meeting, meeting_positions in positions_by_meeting.items():
        meeting_positions.sort()
        if meeting_positions != list(range(len(meeting_positions))):
            raise ValueError(f"positions for meeting {meeting!r} must be contiguous from zero")

    links_by_proposal: dict[str, Mapping[str, Any]] = {}
    for record in links:
        missing = _LINK_FIELDS - set(record)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"link record is missing mandatory fields: {fields}")
        proposal_id = record["proposal_id"]
        if not isinstance(proposal_id, str) or proposal_id not in rows_by_id:
            raise ValueError(f"link points to missing proposal seg_id: {proposal_id!r}")
        if proposal_id in links_by_proposal:
            raise ValueError(f"duplicate link record for proposal {proposal_id!r}")
        if not rows_by_id[proposal_id]["is_layer2_proposal"]:
            raise ValueError(f"link record exists for ineligible proposal {proposal_id!r}")
        if not isinstance(record["status"], str) or record["status"] not in _LINK_STATUSES:
            raise ValueError(f"link record for {proposal_id!r} has invalid status")
        chains = record["responder_chains"]
        if not isinstance(chains, list):
            raise ValueError(f"link record for {proposal_id!r} has invalid responder_chains")
        for chain in chains:
            if not isinstance(chain, list):
                raise ValueError(f"link record for {proposal_id!r} has invalid responder_chains")
            for seg_id in chain:
                if not isinstance(seg_id, str):
                    message = f"link record for {proposal_id!r} has invalid responder_chains"
                    raise ValueError(message)

        if record["has_response"] is not bool(len(chains)):
            raise ValueError(f"has_response inconsistent with responder_chains for {proposal_id!r}")
        if record["status"] == "no_pair" and record["has_response"]:
            raise ValueError(f"no_pair record for {proposal_id!r} cannot have a response")
        if record["status"] == "ok" and not record["has_response"]:
            raise ValueError(f"ok record for {proposal_id!r} must have a response")
        if not isinstance(record["adjacency_tokens"], list):
            raise ValueError(f"link record for {proposal_id!r} has invalid adjacency_tokens")
        for token in record["adjacency_tokens"]:
            if not isinstance(token, str):
                raise ValueError(f"link record for {proposal_id!r} has invalid adjacency_tokens")
        if record["pair_group"] is not None and not isinstance(record["pair_group"], str):
            raise ValueError(f"link record for {proposal_id!r} has invalid pair_group")

        for chain in chains:
            if not chain:
                message = f"link record for {proposal_id!r} contains an empty responder chain"
                raise ValueError(message)
            for response_id in chain:
                if response_id not in rows_by_id:
                    raise ValueError(f"link points to missing response seg_id: {response_id!r}")

            chain_positions = []
            chain_meetings = set()
            for response_id in chain:
                response_row = rows_by_id[response_id]
                chain_positions.append(response_row["position"])
                chain_meetings.add(response_row["meeting"])
            if chain_positions != sorted(chain_positions):
                raise ValueError(f"responder chain for {proposal_id!r} is out of order")
            proposal_meeting = rows_by_id[proposal_id]["meeting"]
            if chain_meetings != {proposal_meeting}:
                raise ValueError(
                    f"responder chain for {proposal_id!r} is out of order or crosses meetings"
                )

        first_response_positions = []
        for chain in chains:
            first_response_positions.append(rows_by_id[chain[0]]["position"])
        if first_response_positions != sorted(first_response_positions):
            raise ValueError(
                f"responder chains for {proposal_id!r} are not ordered by first response position"
            )
        links_by_proposal[proposal_id] = record

    eligible_ids = set()
    for seg_id, row in rows_by_id.items():
        if row["is_layer2_proposal"]:
            eligible_ids.add(seg_id)

    link_ids = set(links_by_proposal)
    if eligible_ids != link_ids:
        missing_links = sorted(eligible_ids - link_ids)
        extra_links = sorted(link_ids - eligible_ids)
        message_parts = []
        if missing_links:
            message_parts.append(
                "proposal rows without link records: " + ", ".join(missing_links)
            )
        if extra_links:
            message_parts.append(
                "link records without eligible rows: " + ", ".join(extra_links)
            )
        message = "; ".join(message_parts)
        raise ValueError(f"Layer-2 proposal/link coverage mismatch ({message})")


def load_release(directory: str | Path) -> ReleaseData:
    """Load, checksum-verify, and validate a complete ``data/v1`` release."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    actual_entries = set()
    for path in root.iterdir():
        actual_entries.add(path.name)
    if actual_entries != RELEASE_FILES:
        missing = sorted(RELEASE_FILES - actual_entries)
        extra = sorted(actual_entries - RELEASE_FILES)
        raise ValueError(
            f"release must contain exactly five files (missing={missing}, extra={extra})"
        )
    _verify_checksums(root)

    rows = _read_jsonl(root / "rows.jsonl")
    links = _read_jsonl(root / "links.jsonl")
    splits = _read_json(root / "splits.json")
    metadata = _read_json(root / "metadata.json")
    if metadata.get("data_version") != "v1":
        raise ValueError("metadata.data_version must be 'v1'")
    validate_release_content(rows, links, splits)
    return ReleaseData(tuple(rows), tuple(links), splits, metadata)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"blank JSONL record at {path.name}:{line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON at {path.name}:{line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}:{line_number} must be a JSON object")
                records.append(value)
    except OSError as error:
        raise ValueError(f"cannot read {path.name}: {error}") from error
    return records


def _verify_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read checksums.sha256: {error}") from error
    expected: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ")
        if len(parts) != 2:
            raise ValueError(f"invalid checksum line: {line!r}")
        digest, filename = parts
        if not _is_sha256(digest):
            raise ValueError(f"invalid checksum line: {line!r}")
        if filename not in _CONTENT_FILES:
            raise ValueError(f"unexpected checksum filename: {filename!r}")
        if filename in expected:
            raise ValueError(f"duplicate checksum entry for {filename!r}")
        expected[filename] = digest
    if set(expected) != set(_CONTENT_FILES):
        raise ValueError("checksums.sha256 must list exactly the four content files")
    filenames = []
    for line in lines:
        _, filename = line.split("  ")
        filenames.append(filename)
    if filenames != sorted(_CONTENT_FILES):
        raise ValueError("checksum entries must be sorted by filename")
    for filename in _CONTENT_FILES:
        actual = _file_sha256(root / filename)
        if actual != expected[filename]:
            raise ValueError(f"checksum mismatch for {filename}")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    for character in value:
        if character not in "0123456789abcdef":
            return False
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
