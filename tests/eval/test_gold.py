"""Tests for src/smi/eval/gold.py and the macro-F1 functions (PR review fixes).

Run from the repo root:   python -m pytest tests/eval/test_gold.py -v
"""

import json

import pytest

from smi.eval.gold import load_gold
from smi.eval.layer1_metrics import macro_f1_6, macro_f1_7
from smi.labels import LABELS


def write_meeting(path, rows: list[tuple[str, str]], extra_lines: list[str] = ()) -> None:
    """Write a meeting file in the data/v4_meetings format: rows are (seg_id, label)."""
    lines = [
        json.dumps({"seg_id": seg_id, "position": i, "speaker": "me001", "text": "ok .", "label": label})
        for i, (seg_id, label) in enumerate(rows)
    ]
    path.write_text("\n".join(lines + list(extra_lines)) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# load_gold
# ---------------------------------------------------------------------------

def test_reads_a_split_folder(tmp_path):
    write_meeting(tmp_path / "M1.jsonl", [("a", "cs"), ("b", "other")])
    write_meeting(tmp_path / "M2.jsonl", [("c", "bk")])
    assert load_gold(tmp_path) == {"a": "cs", "b": "other", "c": "bk"}


def test_blank_lines_are_skipped(tmp_path):
    write_meeting(tmp_path / "M1.jsonl", [("a", "cs")], extra_lines=["", "   "])
    assert load_gold(tmp_path / "M1.jsonl") == {"a": "cs"}


def test_folder_without_jsonl_files_raises(tmp_path):
    # an existing but empty folder must not look like a valid (empty) split
    with pytest.raises(ValueError, match="no \\*.jsonl files"):
        load_gold(tmp_path)


def test_file_with_no_rows_raises(tmp_path):
    (tmp_path / "M1.jsonl").write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows read"):
        load_gold(tmp_path)


def test_invalid_json_names_file_and_line(tmp_path):
    write_meeting(tmp_path / "M1.jsonl", [("a", "cs")], extra_lines=['{"seg_id": "b",'])
    with pytest.raises(ValueError, match="M1.jsonl line 2: invalid JSON"):
        load_gold(tmp_path)


def test_duplicate_seg_id_across_meetings_raises(tmp_path):
    write_meeting(tmp_path / "M1.jsonl", [("a", "cs")])
    write_meeting(tmp_path / "M2.jsonl", [("a", "bk")])
    with pytest.raises(ValueError, match="duplicate seg_id"):
        load_gold(tmp_path)


@pytest.mark.parametrize("line", [
    pytest.param('{"seg_id": "a"}', id="missing-label"),
    pytest.param('{"label": "cs"}', id="missing-seg_id"),
    pytest.param('["a", "cs"]', id="row-is-a-list"),
])
def test_malformed_row_names_file_and_line(tmp_path, line):
    write_meeting(tmp_path / "M1.jsonl", [("a0", "cs")], extra_lines=[line])
    with pytest.raises(ValueError, match="M1.jsonl line 2: expected an object with seg_id and label"):
        load_gold(tmp_path)


def test_seg_id_not_a_string_raises(tmp_path):
    write_meeting(tmp_path / "M1.jsonl", [], extra_lines=['{"seg_id": 7, "label": "cs"}'])
    with pytest.raises(ValueError, match="M1.jsonl line 1: seg_id must be a non-empty string"):
        load_gold(tmp_path)


def test_key_repeated_inside_one_row_raises(tmp_path):
    write_meeting(tmp_path / "M1.jsonl", [], extra_lines=['{"seg_id": "a", "label": "cs", "label": "bk"}'])
    with pytest.raises(ValueError, match="M1.jsonl line 1: duplicate key 'label'"):
        load_gold(tmp_path)


def test_test_split_folder_needs_allow_test(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    write_meeting(test_dir / "M1.jsonl", [("a", "cs")])

    with pytest.raises(ValueError, match="allow_test=True"):  # PROTOCOL §7: never read by accident
        load_gold(test_dir)
    with pytest.raises(ValueError, match="allow_test=True"):  # one meeting file inside test/ counts too
        load_gold(test_dir / "M1.jsonl")
    assert load_gold(test_dir, allow_test=True) == {"a": "cs"}


def test_unknown_label_raises(tmp_path):
    write_meeting(tmp_path / "M1.jsonl", [("a", "aap")])  # raw MRDA tag, not one of the 7 classes
    with pytest.raises(ValueError, match="not in"):
        load_gold(tmp_path)


# ---------------------------------------------------------------------------
# macro-F1: a missing class must raise, never give a wrong number
# ---------------------------------------------------------------------------

def all_classes(f1: float) -> dict[str, dict]:
    return {label: {"p": f1, "r": f1, "f1": f1, "support": 1} for label in LABELS}


@pytest.mark.parametrize("macro", [macro_f1_6, macro_f1_7])
def test_macro_raises_on_missing_class(macro):
    metrics = all_classes(0.5)
    del metrics["cc"]
    with pytest.raises(KeyError):
        macro(metrics)


def test_macro_ignores_extra_keys():
    metrics = all_classes(0.5) | {"unexpected": {"f1": 100.0}}
    assert macro_f1_6(metrics) == pytest.approx(0.5)
    assert macro_f1_7(metrics) == pytest.approx(0.5)
