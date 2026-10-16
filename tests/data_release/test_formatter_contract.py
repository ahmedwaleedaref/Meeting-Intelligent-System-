"""Task 3: ContextFormatter selection semantics, rendering and input validation."""

import pytest

from release_fixtures import formatter_row
from smi.inputs.formatter import ContextFormatter


def stream(count: int = 11, meeting: str = "m1") -> list[dict]:
    return [formatter_row(f"{meeting}-{index}", meeting, index, f"p{index % 3}", f"text{index}") for index in range(count)]


def ids(meeting: str, *positions: int) -> list[str]:
    return [f"{meeting}-{position}" for position in positions]


@pytest.mark.parametrize("k", [0, 1, 3, 5])
def test_left_only_windows_take_k_units_before_the_target(k: int) -> None:
    """Left-only (k, 0) returns the k preceding units and the target."""
    formatter = ContextFormatter(stream())
    assert formatter.select_context("m1-5", k, 0) == ids("m1", *range(5 - k, 6))


@pytest.mark.parametrize("k", [0, 1, 3, 5])
def test_symmetric_windows_are_centered_on_the_target(k: int) -> None:
    """Symmetric (k, k) returns k units on each side."""
    formatter = ContextFormatter(stream())
    assert formatter.select_context("m1-5", k, k) == ids("m1", *range(5 - k, 6 + k))


def test_right_only_and_asymmetric_windows() -> None:
    """Right-only and asymmetric pairs match hand-computed ids."""
    formatter = ContextFormatter(stream())
    assert formatter.select_context("m1-5", 0, 3) == ids("m1", 5, 6, 7, 8)
    assert formatter.select_context("m1-5", 2, 1) == ids("m1", 3, 4, 5, 6)


def test_windows_truncate_at_both_meeting_ends_without_padding() -> None:
    """Fewer available units are returned as is, never padded."""
    formatter = ContextFormatter(stream())
    assert formatter.select_context("m1-1", 5, 0) == ids("m1", 0, 1)
    assert formatter.select_context("m1-9", 0, 5) == ids("m1", 9, 10)
    assert formatter.select_context("m1-0", 3, 3) == ids("m1", 0, 1, 2, 3)


def test_window_never_leaves_the_meeting() -> None:
    """Units from another meeting are never selected."""
    formatter = ContextFormatter([*stream(3, "m1"), *stream(3, "m2")])
    assert formatter.select_context("m1-2", 0, 5) == ids("m1", 2)
    assert formatter.select_context("m2-0", 5, 0) == ids("m2", 0)


def test_input_order_does_not_change_the_stream() -> None:
    """Rows are ordered by position, not by the order they were supplied."""
    shuffled = ContextFormatter(reversed(stream()))
    assert shuffled.select_context("m1-5", 2, 2) == ids("m1", 3, 4, 5, 6, 7)


def test_same_position_in_different_meetings_is_allowed() -> None:
    """Position only has to be unique within a meeting."""
    ContextFormatter([*stream(2, "m1"), *stream(2, "m2")])


def test_raw_mode_uses_speaker_ids_and_joins_lines_with_single_lf() -> None:
    """Raw lines read 'Speaker <id>: <text>', joined by one LF with no trailing LF."""
    text = ContextFormatter(stream()).format("m1-5", 1, 1, "raw")
    assert text == "Speaker p1: text4\nSpeaker p2: [T]text5[/T]\nSpeaker p0: text6"
    assert not text.endswith("\n")


def test_relative_mode_marks_same_and_other_speakers() -> None:
    """Relative mode tags the target's own speaker SAME and everyone else OTHER."""
    text = ContextFormatter(stream()).format("m1-5", 2, 1, "relative")
    assert text.splitlines() == [
        "Speaker OTHER: text3",
        "Speaker OTHER: text4",
        "Speaker SAME: [T]text5[/T]",
        "Speaker OTHER: text6",
    ]


def test_window_letters_follow_first_appearance_and_are_reused() -> None:
    """Letters are handed out left to right per window and reused for repeat speakers."""
    rows = [formatter_row(f"m1-{i}", "m1", i, speaker, f"t{i}") for i, speaker in enumerate(["x", "y", "x", "z", "y"])]
    text = ContextFormatter(rows).format("m1-3", 3, 1, "window_letters")
    assert [line.split(":")[0] for line in text.splitlines()] == [
        "Speaker A", "Speaker B", "Speaker A", "Speaker C", "Speaker B",
    ]


def test_window_letters_continue_past_z() -> None:
    """After Z the letters continue as AA, AB and so on."""
    rows = [formatter_row(f"m1-{i}", "m1", i, f"s{i}", f"t{i}") for i in range(28)]
    lines = ContextFormatter(rows).format("m1-27", 27, 0, "window_letters").splitlines()
    assert lines[0].startswith("Speaker A:") and lines[25].startswith("Speaker Z:")
    assert lines[26].startswith("Speaker AA:") and lines[27].startswith("Speaker AB:")


def test_zero_window_is_identical_across_speaker_modes() -> None:
    """With no context no speaker tag is emitted, so every mode gives the same text."""
    formatter = ContextFormatter(stream())
    outputs = {formatter.format("m1-5", 0, 0, mode) for mode in ("raw", "relative", "window_letters")}
    assert outputs == {"[T]text5[/T]"}


def test_output_never_contains_gold_fields_ids_or_source_ids() -> None:
    """Model-visible text holds only rendered text, markers and speaker tags."""
    formatter = ContextFormatter(stream())
    for mode in ("raw", "relative", "window_letters"):
        text = formatter.format("m1-5", 3, 3, mode)
        for hidden in ("SECRET", "SOURCE-", "m1-5", "label", "fold", "split"):
            assert hidden not in text


def test_repeated_calls_are_identical() -> None:
    """The formatter is a pure function of its rows."""
    formatter = ContextFormatter(stream())
    first = formatter.format("m1-5", 2, 2, "window_letters")
    assert first == formatter.format("m1-5", 2, 2, "window_letters")
    assert formatter.select_context("m1-5", 2, 2) == formatter.select_context("m1-5", 2, 2)


def test_rows_may_be_supplied_as_a_generator() -> None:
    """The constructor accepts any iterable of rows."""
    formatter = ContextFormatter(row for row in stream(3))
    assert formatter.select_context("m1-1", 1, 1) == ids("m1", 0, 1, 2)


@pytest.mark.parametrize("window", [(-1, 0), (0, -1), (1.5, 0), ("1", 0), (True, 0), (0, None)])
def test_invalid_window_sizes_are_rejected(window: tuple) -> None:
    """Window sizes must be non-negative integers."""
    with pytest.raises(ValueError, match="non-negative integer"):
        ContextFormatter(stream()).select_context("m1-5", *window)


def test_unknown_speaker_mode_is_rejected() -> None:
    """An unknown speaker mode raises ValueError."""
    with pytest.raises(ValueError, match="speaker_mode must be one of"):
        ContextFormatter(stream()).format("m1-5", 1, 1, "initials")


def test_formatter_has_no_non_speech_policy() -> None:
    """Rows are removed upstream, so the old nonspeech argument no longer exists."""
    formatter = ContextFormatter(stream())
    with pytest.raises(TypeError):
        formatter.select_context("m1-5", 1, 1, "drop")
    with pytest.raises(TypeError):
        formatter.format("m1-5", 1, 1, "raw", nonspeech="placeholder")


def test_every_row_of_the_meeting_counts_as_a_unit() -> None:
    """With no flags the stream is simply every row of the meeting in position order."""
    formatter = ContextFormatter(stream(6))
    assert formatter.select_context("m1-3", 10, 10) == ids("m1", 0, 1, 2, 3, 4, 5)


def test_rows_need_no_flag_fields() -> None:
    """Rows without any non-speech flag fields are accepted as they are."""
    row = formatter_row("m1-0", "m1", 0, "p0", "hello")
    assert not {"is_nonspeech", "is_nonlabeled", "is_empty"} & set(row)
    assert ContextFormatter([row]).format("m1-0", 0, 0, "raw") == "[T]hello[/T]"


def test_unknown_target_is_rejected() -> None:
    """A target that is not in the snapshot raises KeyError."""
    with pytest.raises(KeyError, match="unknown seg_id"):
        ContextFormatter(stream()).select_context("nope", 1, 1)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate seg_id"),
        (lambda rows: rows[1].update(position=0), "duplicate position"),
        (lambda rows: rows[0].update(seg_id=""), "non-empty seg_id"),
        (lambda rows: rows[0].update(meeting=""), "non-empty meeting"),
        (lambda rows: rows[0].update(position=-1), "non-negative integer position"),
        (lambda rows: rows[0].update(position=True), "non-negative integer position"),
        (lambda rows: rows[0].update(speaker_id=""), "non-empty speaker_id"),
        (lambda rows: rows[0].update(text=None), "text as a string"),
    ],
)
def test_malformed_rows_are_rejected_at_construction(mutate, message: str) -> None:
    """Bad ids, positions, speakers, text or flags are rejected up front."""
    rows = stream(3)
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        ContextFormatter(rows)
