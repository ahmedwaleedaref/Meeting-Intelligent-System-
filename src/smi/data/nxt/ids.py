"""Stable identifiers for parsed NXT dialogue-act rows."""

from decimal import Decimal, InvalidOperation


def milliseconds_from_source_time(value: str) -> int:
    """Convert an NXT decimal time to milliseconds without float rounding."""
    try:
        decimal_value = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Invalid NXT time: {value!r}") from error

    milliseconds = decimal_value * 1000
    if milliseconds != milliseconds.to_integral_value():
        raise ValueError(f"NXT time has more than millisecond precision: {value!r}")
    return int(milliseconds)


def make_seg_id(
    meeting: str,
    channel: str,
    source_start: str,
    source_end: str,
    part_index: int,
    number_of_parts: int,
) -> str:
    """Build D1's ID from source timing, never from derived part timing."""
    start_ms = milliseconds_from_source_time(source_start)
    end_ms = milliseconds_from_source_time(source_end)
    segment_id = f"{meeting}-{channel}_{start_ms:07d}_{end_ms:07d}"
    if number_of_parts > 1:
        segment_id += f"_p{part_index + 1}"
    return segment_id
