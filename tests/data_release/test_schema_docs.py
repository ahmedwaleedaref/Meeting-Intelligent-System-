"""Part A: schema docs, concept notes and the task README stay consistent with the code."""

import json
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
    """Schema 03 names the five files, the target markers, all speaker modes, cleaned input and position gaps."""
    text = (SCHEMA / "03_release.md").read_text(encoding="utf-8")
    for token in (
        "rows.jsonl", "links.jsonl", "splits.json", "metadata.json", "checksums.sha256",
        "[T]", "[/T]", "raw", "relative", "window_letters",
        "select_context", "source_sha256", "malformed", "ambiguous", "cleaned", "gaps", "renumbers",
    ):
        assert token in text, token
    assert 'nonspeech="' not in text and "has no non-speech policy" in text


@pytest.mark.parametrize("path", sorted((TASK_FOLDER / "provenance").glob("*.json")), ids=lambda path: path.name)
def test_every_provenance_file_is_valid_json(path: Path) -> None:
    """Each provenance file parses as a JSON object, so a merge slip cannot hide in one."""
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


def test_concept_note_for_nxt_exists() -> None:
    """The NXT concept note required from Task 1 is present."""
    assert (CONCEPTS / "nxt-icsi-annotations.md").read_text(encoding="utf-8").strip()


def test_task_readme_has_the_documented_sections() -> None:
    """The README has the overview, per-task, deviations, decisions, usage and limitations sections."""
    text = (TASK_FOLDER / "README.md").read_text(encoding="utf-8")
    headings = {re.sub(r"^\d+\.\s*", "", heading).lower() for heading in re.findall(r"^## (.+)$", text, re.MULTILINE)}
    assert {
        "overview",
        "what each task did",
        "deviations from the original plan",
        "decisions and choices",
        "how to use the pipeline",
        "limitations and open items",
    } <= headings


def test_summary_stays_within_the_team_length_limit() -> None:
    """SUMMARY.md exists and has at most 60 non-blank lines."""
    lines = (TASK_FOLDER / "SUMMARY.md").read_text(encoding="utf-8").splitlines()
    assert 0 < sum(1 for line in lines if line.strip()) <= 60


@pytest.mark.parametrize("name", ["README.md", "SUMMARY.md"])
def test_relative_links_in_the_task_docs_resolve(name: str) -> None:
    """Every relative link in the task README and summary points at a file that exists."""
    text = (TASK_FOLDER / name).read_text(encoding="utf-8")
    missing = [
        target
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text)
        if not target.startswith("http") and not (TASK_FOLDER / target).exists()
    ]
    assert missing == []


def test_schema_02_documents_every_annotation_and_link_field() -> None:
    """Schema 02 lists every annotation field and every link field, in the order the code writes them."""
    documented = table_fields(SCHEMA / "02_annotations_links.md")
    assert documented == [*ANNOTATION_FIELDS, *LinkRecord.__annotations__]


def test_concept_note_for_mrda_tags_and_adjacency_pairs_exists() -> None:
    """The MRDA tag and adjacency-pair concept note required from Task 2 is present."""
    assert (CONCEPTS / "mrda-tags-and-adjacency-pairs.md").read_text(encoding="utf-8").strip()
