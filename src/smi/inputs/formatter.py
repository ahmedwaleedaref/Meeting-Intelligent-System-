"""Deterministic context selection and rendering for frozen C3 rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


_SPEAKER_MODES = {"raw", "relative", "window_letters"}
_NONSPEECH_POLICIES = {"drop", "placeholder"}


@dataclass(frozen=True)
class _Unit:
    seg_id: str
    meeting: str
    position: int
    speaker_id: str
    text: str
    flagged: bool


class ContextFormatter:
    """Format context from an immutable snapshot of C3 dialogue-act rows.

    Rows are ordered within each meeting by ``position``. The constructor keeps
    only fields needed for selection and rendering, so gold annotations and
    split metadata cannot enter model-visible text.
    """

    def __init__(self, rows: Iterable[Mapping[str, object]]) -> None:
        units: list[_Unit] = []
        seen_ids: set[str] = set()
        seen_positions: set[tuple[str, int]] = set()

        for row in rows:
            seg_id = row.get("seg_id")
            meeting = row.get("meeting")
            position = row.get("position")
            speaker_id = row.get("speaker_id")
            text = row.get("text")
            is_nonspeech = row.get("is_nonspeech")
            is_empty = row.get("is_empty")

            if not isinstance(seg_id, str) or not seg_id:
                raise ValueError("every row must have a non-empty seg_id")
            if seg_id in seen_ids:
                raise ValueError(f"duplicate seg_id: {seg_id!r}")
            if not isinstance(meeting, str) or not meeting:
                raise ValueError(f"row {seg_id!r} must have a non-empty meeting")
            if type(position) is not int or position < 0:
                raise ValueError(f"row {seg_id!r} must have a non-negative integer position")
            if (meeting, position) in seen_positions:
                raise ValueError(f"duplicate position {position} in meeting {meeting!r}")
            if not isinstance(speaker_id, str) or not speaker_id:
                raise ValueError(f"row {seg_id!r} must have a non-empty speaker_id")
            if not isinstance(text, str):
                raise ValueError(f"row {seg_id!r} must have text as a string")
            if type(is_nonspeech) is not bool or type(is_empty) is not bool:
                raise ValueError(
                    f"row {seg_id!r} must have boolean is_nonspeech and is_empty fields"
                )

            seen_ids.add(seg_id)
            seen_positions.add((meeting, position))
            units.append(
                _Unit(
                    seg_id=seg_id,
                    meeting=meeting,
                    position=position,
                    speaker_id=speaker_id,
                    text=text,
                    flagged=is_nonspeech or is_empty,
                )
            )

        by_meeting: dict[str, list[_Unit]] = {}
        by_id: dict[str, _Unit] = {}
        for unit in units:
            if unit.meeting not in by_meeting:
                by_meeting[unit.meeting] = []
            by_meeting[unit.meeting].append(unit)
            by_id[unit.seg_id] = unit

        self._by_meeting = {}
        for meeting, meeting_units in by_meeting.items():
            meeting_units.sort(key=lambda unit: unit.position)
            self._by_meeting[meeting] = tuple(meeting_units)
        self._by_id = by_id

    def select_context(
        self,
        seg_id: str,
        k_left: int,
        k_right: int,
        nonspeech: str = "drop",
    ) -> list[str]:
        """Return selected unit IDs in stream order, always including the target."""
        self._validate_window(k_left, k_right, nonspeech)
        target = self._get_target(seg_id)
        meeting_units = self._by_meeting[target.meeting]

        if nonspeech == "drop":
            stream = []
            for unit in meeting_units:
                if not unit.flagged or unit.seg_id == seg_id:
                    stream.append(unit)
        else:
            stream = meeting_units

        target_index = 0
        for index, unit in enumerate(stream):
            if unit.seg_id == seg_id:
                target_index = index
                break

        start = max(0, target_index - k_left)
        stop = min(len(stream), target_index + k_right + 1)
        selected_ids = []
        for unit in stream[start:stop]:
            selected_ids.append(unit.seg_id)
        return selected_ids

    def format(
        self,
        seg_id: str,
        k_left: int,
        k_right: int,
        speaker_mode: str,
        *,
        nonspeech: str = "drop",
    ) -> str:
        """Render a context window without exposing annotations or split fields."""
        if speaker_mode not in _SPEAKER_MODES:
            raise ValueError(f"speaker_mode must be one of {sorted(_SPEAKER_MODES)}")

        selected_ids = self.select_context(seg_id, k_left, k_right, nonspeech)
        selected = []
        for item_id in selected_ids:
            selected.append(self._by_id[item_id])
        target = self._get_target(seg_id)
        emit_speakers = bool(k_left or k_right)

        first_seen_speakers: dict[str, str] = {}
        lines = []
        for unit in selected:
            content = unit.text
            if nonspeech == "placeholder" and unit.flagged:
                content = "[nonverbal]"
            if unit.seg_id == seg_id:
                content = f"[T]{content}[/T]"

            if emit_speakers:
                if speaker_mode == "raw":
                    speaker_tag = unit.speaker_id
                elif speaker_mode == "relative":
                    if unit.speaker_id == target.speaker_id:
                        speaker_tag = "SAME"
                    else:
                        speaker_tag = "OTHER"
                else:
                    if unit.speaker_id not in first_seen_speakers:
                        letter_index = len(first_seen_speakers)
                        first_seen_speakers[unit.speaker_id] = self._window_letter(letter_index)
                    speaker_tag = first_seen_speakers[unit.speaker_id]
                lines.append(f"Speaker {speaker_tag}: {content}")
            else:
                lines.append(content)
        return "\n".join(lines)

    def _get_target(self, seg_id: str) -> _Unit:
        try:
            return self._by_id[seg_id]
        except KeyError as error:
            raise KeyError(f"unknown seg_id: {seg_id!r}") from error

    @staticmethod
    def _validate_window(k_left: int, k_right: int, nonspeech: str) -> None:
        for name, value in (("k_left", k_left), ("k_right", k_right)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if nonspeech not in _NONSPEECH_POLICIES:
            raise ValueError(f"nonspeech must be one of {sorted(_NONSPEECH_POLICIES)}")

    @staticmethod
    def _window_letter(index: int) -> str:
        """Return spreadsheet-style letters: A..Z, AA, AB, ..."""
        result = ""
        while index >= 0:
            index, remainder = divmod(index, 26)
            result = chr(ord("A") + remainder) + result
            index -= 1
        return result
