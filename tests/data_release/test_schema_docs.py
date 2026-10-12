"""Part A: schema docs, concept notes and the task README stay consistent with the code."""

import re
from pathlib import Path

import pytest

from smi.data.annotate.records import ANNOTATION_FIELDS, LinkRecord
from smi.data.nxt.validation import C1_FIELDS, ERROR_FLAGS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_FOLDER = PROJECT_ROOT / "tasks" / "A-data-transformation"
SCHEMA = TASK_FOLDER / "schema"
CONCEPTS = PROJECT_ROOT / "concepts"
TIMING_METHODS = {"element", "transcript_word_boundary", "transcript_proportional", "source_proportional"}


def table_fields(path: Path) -> list[str]:
    """Return the backticked field names in the first column of every table row."""
    fields: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `"):
            first_cell = line.split("|")[1]
            fields.extend(re.findall(r"`([a-z_0-9]+)`", first_cell))
    return fields


def backticked(text: str) -> set[str]:
    return set(re.findall(r"`([^`]+)`", text))


def test_schema_01_documents_exactly_the_c1_fields_in_order() -> None:
    """Schema 01 lists every C1 field once, in the order the parser emits them."""
    assert table_fields(SCHEMA / "01_parsed_rows.md") == list(C1_FIELDS)


def test_schema_01_lists_the_timing_methods_the_parser_can_emit() -> None:
    """The four timing methods in the code are all described in schema 01."""
    section = (SCHEMA / "01_parsed_rows.md").read_text(encoding="utf-8").split("## Timing methods")[1].split("##")[0]
    assert {item for item in backticked(section) if item in TIMING_METHODS} == TIMING_METHODS


def test_schema_01_quality_flag_vocabulary_matches_the_code() -> None:
    """Schema 01 names the same error-class and informational flags as the parser."""
    section = (SCHEMA / "01_parsed_rows.md").read_text(encoding="utf-8").split("## Quality flags")[1].split("##")[0]
    informational, error_class = section.split("Error-class:")
    assert backticked(error_class) >= ERROR_FLAGS
    assert {"no_word_ref", "type_fallback_original", "word_time_missing"} <= backticked(informational)
    assert not ERROR_FLAGS & backticked(informational)


def test_schema_03_describes_the_release_and_the_formatter_contract() -> None:
    """Schema 03 names the five files, the target markers, the placeholder and all speaker modes."""
    text = (SCHEMA / "03_release.md").read_text(encoding="utf-8")
    for token in (
        "rows.jsonl", "links.jsonl", "splits.json", "metadata.json", "checksums.sha256",
        "[T]", "[/T]", "[nonverbal]", "raw", "relative", "window_letters",
        "select_context", "source_sha256", "malformed", "ambiguous",
    ):
        assert token in text, token


def test_concept_note_for_nxt_exists() -> None:
    """The NXT concept note required from Task 1 is present."""
    assert (CONCEPTS / "nxt-icsi-annotations.md").read_text(encoding="utf-8").strip()


def test_task_readme_has_goal_status_and_decisions() -> None:
    """The task README has the Goal, Status and Decisions sections."""
    headings = re.findall(r"^## (.+)$", (TASK_FOLDER / "README.md").read_text(encoding="utf-8"), re.MULTILINE)
    assert {"Goal", "Status", "Decisions"} <= set(headings)


@pytest.mark.xfail(strict=True, reason="schema/02_annotations_links.md is not on this branch yet")
def test_schema_02_documents_every_annotation_and_link_field() -> None:
    """Schema 02 lists every annotation field and every link field."""
    documented = table_fields(SCHEMA / "02_annotations_links.md")
    assert set(documented) == set(ANNOTATION_FIELDS) | set(LinkRecord.__annotations__)


@pytest.mark.xfail(strict=True, reason="concepts/mrda-tags-and-adjacency-pairs.md is not on this branch yet")
def test_concept_note_for_mrda_tags_and_adjacency_pairs_exists() -> None:
    """The MRDA tag and adjacency-pair concept note required from Task 2 is present."""
    assert (CONCEPTS / "mrda-tags-and-adjacency-pairs.md").read_text(encoding="utf-8").strip()
