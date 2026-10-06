"""Readable, deterministic parser for the ICSI Core NXT C1 artifact."""

from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ElementTree

from .ids import make_seg_id, milliseconds_from_source_time
from .ordering import assign_positions
from .validation import C1_FIELDS, ERROR_FLAGS


NITE_ID = "{http://nite.sourceforge.net/}id"
WORD_KINDS = {
    "w",
    "disfmarker",
    "vocalsound",
    "nonvocalsound",
    "comment",
    "pause",
}
# NXT uses both a single `id(...)` and an inclusive `id(...)..id(...)` range.
REFERENCE_PATTERN = re.compile(
    r"^(?P<file>[^#]+)#id\((?P<first>[^)]+)\)(?:\.\.id\((?P<last>[^)]+)\))?$"
)


def local_name(xml_tag: str) -> str:
    """Return an XML tag without its optional namespace."""
    return xml_tag.rsplit("}", 1)[-1]


def parse_time(value: str | None) -> float | None:
    """Parse a required NXT decimal time, returning None when it is unusable."""
    if value is None or not value.strip():
        return None
    try:
        return float(Decimal(value))
    except (InvalidOperation, ValueError):
        return None


def normalize_lookup_id(value: str) -> str:
    """Normalize comma-formatted IDs for lookup only; stored source IDs stay untouched."""
    return value.replace(",", "")


def word_text(element: ElementTree.Element) -> str:
    """Use element text, or NXT's description for non-verbal elements without text."""
    text = (element.text or "").strip()
    if text:
        return text
    return element.get("description", "")


def load_words(words_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read one words XML file in document order and build an ID lookup."""
    root = ElementTree.parse(words_path).getroot()
    words: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}
    for element in root:
        kind = local_name(element.tag)
        if kind not in WORD_KINDS:
            continue
        source_id = element.get(NITE_ID)
        if not source_id:
            continue
        index_by_id[normalize_lookup_id(source_id)] = len(words)
        words.append(
            {
                "text": word_text(element),
                "kind": kind,
                "start": parse_time(element.get("starttime")),
                "end": parse_time(element.get("endtime")),
            }
        )
    return words, index_by_id


def resolve_word_range(
    child: ElementTree.Element | None,
    words: list[dict[str, Any]],
    index_by_id: dict[str, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve one NITE child range without concealing bad source references."""
    if child is None:
        return [], ["no_word_ref"]
    href = child.get("href")
    match = REFERENCE_PATTERN.fullmatch(href or "")
    if match is None:
        return [], ["ref_malformed"]
    first = index_by_id.get(normalize_lookup_id(match.group("first")))
    last_id = match.group("last") or match.group("first")
    last = index_by_id.get(normalize_lookup_id(last_id))
    if first is None or last is None:
        return [], ["ref_unresolved"]
    if first > last:
        return [], ["ref_unresolved"]
    selected_words = [dict(word) for word in words[first : last + 1]]
    flags: list[str] = []
    if any(word["start"] is None or word["end"] is None for word in selected_words):
        flags.append("word_time_missing")
    return selected_words, flags


def load_transcripts(
    transcript_directory: Path,
    corrections: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Map transcript IDs to both available pipe-preserving transcript renderings."""
    records: dict[str, list[str]] = {}
    for path in sorted(transcript_directory.glob("*.trans")):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split(",", 2)
            if len(fields) >= 2:
                # The first rendering is the transcript text and the optional second
                # rendering is closer to lexical tokens in NXT (for example o_k/OK).
                records[fields[0]] = [field for field in fields[1:] if "|" in field]
    if corrections:
        records.update(corrections)
    return records


def transcript_key(meeting: str, channel: str, source_start: str, source_end: str) -> str:
    """Build the transcript key from exact source millisecond timestamps."""
    transcript_channel = channel.lower()
    # A handful of NXT channels use a `2c0` spelling while the transcript uses c0.
    if transcript_channel.startswith("2c"):
        transcript_channel = transcript_channel[1:]
    return (
        f"{meeting}-{transcript_channel}_{milliseconds_from_source_time(source_start):07d}_"
        f"{milliseconds_from_source_time(source_end):07d}"
    )


def alignment_text(value: str) -> str:
    """Normalization used only to compare transcript and NXT lexical content."""
    return "".join(
        character.lower()
        for character in value
        if character.isalnum()
    )


def find_pipe_boundaries(
    words: list[dict[str, Any]],
    transcript_text: str,
    number_of_parts: int,
) -> list[int] | None:
    """Find every transcript pipe as a unique boundary in the ordered NXT words."""
    transcript_parts = transcript_text.split("|")
    if len(transcript_parts) != number_of_parts:
        return None
    boundaries: list[int] = []
    consumed = ""
    for part_text in transcript_parts[:-1]:
        consumed += alignment_text(part_text)
        matching_indexes: list[int] = []
        source_prefix = ""
        for index, word in enumerate(words, start=1):
            # Transcript words correspond to lexical `w` elements. NXT's other
            # element kinds remain in C1 text, but descriptions such as "breath"
            # are not transcript words and would make a real boundary look wrong.
            if word["kind"] == "w":
                source_prefix += alignment_text(word["text"])
            if source_prefix == consumed:
                matching_indexes.append(index)
        if not matching_indexes:
            return None
        # A punctuation-only NXT token does not change normalized text. It belongs
        # to the preceding transcript portion, so select the last equivalent index.
        boundaries.append(matching_indexes[-1])
    if boundaries != sorted(boundaries):
        return None
    return boundaries


def split_words(
    words: list[dict[str, Any]],
    boundaries: list[int],
    number_of_parts: int,
) -> list[list[dict[str, Any]]]:
    """Split an ordered word range at already-validated boundary indexes."""
    endpoints = [0, *boundaries, len(words)]
    return [
        words[endpoints[index] : endpoints[index + 1]]
        for index in range(number_of_parts)
    ]


def derive_part_times(
    source_start: float,
    source_end: float,
    parts: list[list[dict[str, Any]]],
) -> tuple[list[tuple[float, float]], str]:
    """Derive non-overlapping split times from word boundaries or proportional fallback."""
    if len(parts) == 1:
        return [(source_start, source_end)], "element"

    boundary_times: list[float] = []
    all_boundaries_have_word_time = True
    for left, right in zip(parts, parts[1:]):
        left_end = left[-1]["end"] if left else None
        right_start = right[0]["start"] if right else None
        if left_end is not None:
            boundary_times.append(left_end)
        elif right_start is not None:
            boundary_times.append(right_start)
        else:
            all_boundaries_have_word_time = False
            break
    if not all_boundaries_have_word_time:
        total_words = max(sum(len(part) for part in parts), 1)
        words_before = 0
        boundary_times = []
        for part in parts[:-1]:
            words_before += len(part)
            fraction = words_before / total_words
            boundary_times.append(source_start + (source_end - source_start) * fraction)
        method = "transcript_proportional"
    else:
        boundary_times = [
            min(source_end, max(source_start, value))
            for value in boundary_times
        ]
        method = "transcript_word_boundary"

    endpoints = [source_start, *boundary_times, source_end]
    timings = [
        (endpoints[index], endpoints[index + 1])
        for index in range(len(parts))
    ]
    return timings, method


def build_text(words: list[dict[str, Any]]) -> str:
    """Keep every source token and join non-empty token text with single spaces."""
    return " ".join(word["text"] for word in words if word["text"])


def parse_corpus(
    raw_root: Path,
    transcript_corrections: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, str]]]:
    """Parse the release into C1 rows, measured counts, and visible split failures."""
    dialogue_directory = raw_root / "DialogueActs"
    words_directory = raw_root / "Words"
    transcript_directory = raw_root / "transcripts"
    if (
        not dialogue_directory.is_dir()
        or not words_directory.is_dir()
        or not transcript_directory.is_dir()
    ):
        raise FileNotFoundError(
            "Expected DialogueActs, Words, and transcripts under the raw-data root"
        )

    transcripts = load_transcripts(transcript_directory, transcript_corrections)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    transcript_problems: list[dict[str, str]] = []
    word_cache: dict[str, tuple[list[dict[str, Any]], dict[str, int]]] = {}
    da_paths = sorted(dialogue_directory.glob("*.dialogue-acts.xml"))
    counts["dialogue_act_files"] = len(da_paths)
    counts["meetings"] = len({path.name.split(".")[0] for path in da_paths})

    for da_path in da_paths:
        meeting, agent, _ = da_path.name.split(".", 2)
        root = ElementTree.parse(da_path).getroot()

        for element_ordinal, dialogue_act in enumerate(root.findall("dialogueact")):
            counts["dialogue_act_elements"] += 1

            # Read and validate the source dialogue-act time span.
            source_start_raw = dialogue_act.get("starttime")
            source_end_raw = dialogue_act.get("endtime")
            source_start = parse_time(source_start_raw)
            source_end = parse_time(source_end_raw)
            flags: list[str] = []

            if (
                source_start is None
                or source_end is None
                or source_start > source_end
            ):
                flags.append("time_invalid")
                source_start = source_start if source_start is not None else 0.0
                source_end = (
                    source_end
                    if source_end is not None and source_end >= source_start
                    else source_start
                )
                source_start_raw = source_start_raw or "0"
                source_end_raw = source_end_raw or source_start_raw

            assert source_start_raw is not None
            assert source_end_raw is not None

            # Select the source type, falling back only when the original type
            # is available.
            raw_type = dialogue_act.get("type")
            original_type = dialogue_act.get("original-type")
            selected_type = raw_type

            if not selected_type:
                if original_type:
                    selected_type = original_type
                    flags.append("type_fallback_original")
                else:
                    selected_type = ""
                    flags.append("type_unavailable")

            part_types = selected_type.split("|")
            counts["independent_pipe_parts"] += len(part_types)

            # Resolve the referenced words, reusing each words file after its
            # first load.
            child = next(
                (child for child in dialogue_act if local_name(child.tag) == "child"),
                None,
            )
            href = child.get("href") if child is not None else ""
            word_file_name = (
                href.split("#", 1)[0] if href else f"{meeting}.{agent}.words.xml"
            )
            words_path = words_directory / word_file_name
            if words_path.exists() and word_file_name not in word_cache:
                word_cache[word_file_name] = load_words(words_path)
            word_data = word_cache.get(word_file_name, ([], {}))
            resolved_words, word_flags = resolve_word_range(child, *word_data)
            flags.extend(word_flags)

            # Split pipe-delimited source types and derive each part's timing.
            part_words: list[list[dict[str, Any]]]
            timing_method: str

            if len(part_types) == 1:
                part_words = [resolved_words]
                timings, timing_method = derive_part_times(
                    source_start,
                    source_end,
                    part_words,
                )
            elif "no_word_ref" in flags:
                # A missing reference is a valid NXT condition. There is no word
                # range to align, so keep every part empty and divide only its
                # source span; do not mislabel this as an unresolved reference.
                part_words = [[] for _ in part_types]
                timings, _ = derive_part_times(source_start, source_end, part_words)
                timing_method = "source_proportional"

            else:
                try:
                    key = transcript_key(
                        meeting,
                        dialogue_act.get("channel", ""),
                        source_start_raw,
                        source_end_raw,
                    )
                except ValueError:
                    key = ""
                transcript_texts = transcripts.get(key, [])
                boundaries: list[int] | None = None
                for transcript_text in transcript_texts:
                    candidate_boundaries = find_pipe_boundaries(
                        resolved_words,
                        transcript_text,
                        len(part_types),
                    )
                    if candidate_boundaries is not None:
                        boundaries = candidate_boundaries
                        break
                if boundaries is None:
                    # The C1 vocabulary has no separate pipe flag. ref_unresolved is the
                    # contract's error state for source data that cannot be resolved.
                    if "ref_unresolved" not in flags:
                        flags.append("ref_unresolved")
                    part_words = [[] for _ in part_types]
                    timings, timing_method = derive_part_times(
                        source_start,
                        source_end,
                        part_words,
                    )
                    transcript_problems.append(
                        {
                            "source_da_id": dialogue_act.get(NITE_ID, "<missing id>"),
                            "reason": (
                                "no pipe-preserving transcript record"
                                if not transcript_texts
                                else "transcript wording did not align uniquely"
                            ),
                            "context": (
                                f"transcript key={key or '<invalid time>'}; "
                                f"record_count={len(transcript_texts)}"
                            ),
                        }
                    )
                else:
                    part_words = split_words(resolved_words, boundaries, len(part_types))
                    timings, timing_method = derive_part_times(
                        source_start,
                        source_end,
                        part_words,
                    )

            for part_index, part_type in enumerate(part_types):
                words_for_part = part_words[part_index]

                try:
                    segment_id = make_seg_id(
                        meeting,
                        dialogue_act.get("channel", ""),
                        source_start_raw,
                        source_end_raw,
                        part_index,
                        len(part_types),
                    )
                except ValueError:
                    segment_id = (
                        f"{meeting}-{dialogue_act.get('channel', '')}"
                        "_0000000_0000000"
                    )
                    if len(part_types) > 1:
                        segment_id += f"_p{part_index + 1}"

                row = {
                    "seg_id": segment_id,
                    "meeting": meeting,
                    "agent": agent,
                    "channel": dialogue_act.get("channel", ""),
                    "speaker_id": dialogue_act.get("participant", ""),
                    "src_start": source_start,
                    "src_end": source_end,
                    "start": timings[part_index][0],
                    "end": timings[part_index][1],
                    "timing_method": timing_method,
                    "position": -1,
                    "source_da_id": dialogue_act.get(NITE_ID, ""),
                    "element_ordinal": element_ordinal,
                    "part_index": part_index,
                    "n_parts": len(part_types),
                    "raw_type": raw_type,
                    "original_type": original_type,
                    "part_type": part_type,
                    "adjacency_raw": dialogue_act.get("adjacency"),
                    "comment": dialogue_act.get("comment"),
                    "text": build_text(words_for_part),
                    "words": words_for_part,
                    "quality_flags": list(dict.fromkeys(flags)),
                }
                assert tuple(row) == C1_FIELDS
                rows.append(row)
    return assign_positions(rows), counts, transcript_problems


def has_fatal_errors(
    rows: list[dict[str, Any]],
    transcript_problems: list[dict[str, str]],
    contract_problems: list[str],
) -> bool:
    """State whether output is unsafe to hand to downstream frozen-data tasks."""
    return bool(
        contract_problems
        or transcript_problems
        or any(ERROR_FLAGS.intersection(row["quality_flags"]) for row in rows)
    )
