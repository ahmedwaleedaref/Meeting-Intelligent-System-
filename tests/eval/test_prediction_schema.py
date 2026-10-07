"""Tests for src/smi/eval/prediction_schema.py.

Every validation rule in tasks/B-eval-harness/task.md (section B1, "Validation") is tested:
a broken prediction must raise PredictionFileError, a correct one must load unchanged.

Run from the repo root:   python -m pytest tests/eval -v
"""

import json  

import pytest  

from smi.eval.prediction_schema import ( 
    PredictionFileError, 
    load_predictions,  # reads a whole .jsonl file and returns {seg_id: {"label", "scores"}}
    validate_one_parsed_json,  # checks one line that is already converted to a Python dict
)
from smi.labels import LABELS  

# ---------------------------------------------------------------------------
# Helpers: each one returns a FRESH valid object, so every test changes exactly one thing
# ---------------------------------------------------------------------------

def valid_scores() -> dict:
    """Valid scores: all 7 labels, each in [0, 1], summing to 1. `cs` is the highest."""
    return {"cs": 0.70, "co": 0.05, "aa": 0.05, "bk": 0.05, "ar": 0.05, "cc": 0.05, "other": 0.05}


def scores_from(**values) -> dict:
    """All 7 labels set to 0.0, then the given labels overwritten, e.g. scores_from(cs=1.0)."""
    scores = {label: 0.0 for label in LABELS}  # start with every label at 0.0
    scores.update(values)  # overwrite only the labels the test passes in
    return scores


def valid_record(**changes) -> dict:
    """One valid prediction line as a dict; keyword arguments replace or add fields."""
    record = {"seg_id": "Bdb001-c3_0832170_0832370", "label": "cs", "scores": valid_scores()}  # a real seg_id format
    record.update(changes)  # e.g. valid_record(label="xyz") breaks only the label
    return record


def write_jsonl(tmp_path, lines: list[str]):
    """Write text lines to tmp_path/predictions.jsonl and return the file path."""
    path = tmp_path / "predictions.jsonl"  # tmp_path is a fresh empty folder pytest gives each test
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")  # one line per item, UTF-8 like real files
    return path


# ---------------------------------------------------------------------------
# validate_one_parsed_json: lines that MUST pass
# ---------------------------------------------------------------------------

def test_valid_record_with_scores_passes():
    validate_one_parsed_json(valid_record())  # no exception raised = the test passes


def test_valid_record_with_null_scores_passes():
    validate_one_parsed_json(valid_record(scores=None))  # null scores are allowed (generation-only prompting)


def test_one_hot_integer_scores_pass():
    # Decision: integer 0 and 1 count as valid numbers, because a model may output one-hot scores.
    validate_one_parsed_json(valid_record(scores=scores_from(cs=1, co=0, aa=0, bk=0, ar=0, cc=0, other=0)))


def test_scores_sum_inside_tolerance_passes():
    validate_one_parsed_json(valid_record(scores=valid_scores() | {"cs": 0.7005}))  # sum = 1.0005, inside 1 ± 0.001


# ---------------------------------------------------------------------------
# validate_one_parsed_json: lines that MUST fail
# Each case breaks exactly ONE rule, and `message` is a phrase from the error the code must raise.
# Decision: we check the message too, so a test only passes when the RIGHT rule caught the
# problem (e.g. a bad score must be caught by the range check, not by luck by the sum check).
# ---------------------------------------------------------------------------

BAD_RECORDS = [
    # Rule 1: the line must be a JSON object with exactly the keys seg_id, label, scores
    pytest.param(["Bdb001-c3_0832170_0832370", "cs", None], "expected a JSON object", id="line-is-a-list"),
    pytest.param("cs", "expected a JSON object", id="line-is-a-string"),
    pytest.param({"seg_id": "Bdb001-c3_0832170_0832370", "label": "cs"}, "wrong keys", id="missing-scores-key"),
    pytest.param(valid_record(gold="cs"), "wrong keys", id="extra-gold-key"),  
    # Rule 2: seg_id and label are non-empty strings; scores is an object or null
    pytest.param(valid_record(seg_id=123), "seg_id must be", id="seg_id-is-a-number"),
    pytest.param(valid_record(seg_id=""), "seg_id must be", id="seg_id-is-empty"),
    pytest.param(valid_record(label=None), "label must be", id="label-is-null"),
    pytest.param(valid_record(label=""), "label must be", id="label-is-empty"),
    pytest.param(valid_record(scores=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]), "scores must be", id="scores-is-a-list"),
    pytest.param(valid_record(scores="high"), "scores must be", id="scores-is-a-string"),
    # Rule 3: label is one of the 7 classes
    pytest.param(valid_record(label="xyz"), "not in", id="label-unknown"),
    pytest.param(valid_record(label="CS"), "not in", id="label-wrong-case"),
    pytest.param(valid_record(label="aap"), "not in", id="label-is-raw-mrda-tag"),  
    # Rule 4a: scores has exactly the 7 label keys
    pytest.param(valid_record(scores={k: v for k, v in valid_scores().items() if k != "cc"}),
                 "scores has wrong keys", id="scores-missing-a-label"),
    pytest.param(valid_record(scores=valid_scores() | {"maybe": 0.0}), "scores has wrong keys", id="scores-extra-label"),
    # Rule 4b: every score is a number in [0, 1]  (each case still sums to 1, so only the range check can catch it)
    pytest.param(valid_record(scores=scores_from(cs=1.1, co=-0.1)), "must be a number", id="score-above-1"),
    pytest.param(valid_record(scores=scores_from(cs=0.8, co=-0.1, aa=0.3)), "must be a number", id="score-negative"),
    pytest.param(valid_record(scores=scores_from(cs=True)), "must be a number", id="score-is-boolean"), 
    pytest.param(valid_record(scores=scores_from(cs="1.0")), "must be a number", id="score-is-a-string"),
    pytest.param(valid_record(scores=scores_from(cs=None, co=1.0)), "must be a number", id="score-is-null"),
    pytest.param(valid_record(scores=scores_from(cs=float("nan"), co=1.0)), "must be a number", id="score-is-nan"),
    pytest.param(valid_record(scores=scores_from(cs=float("inf"))), "must be a number", id="score-is-infinity"),
    # Rule 4c: the scores sum to 1 within ± 0.001
    pytest.param(valid_record(scores=valid_scores() | {"cs": 0.60}), "scores sum to", id="sum-too-low"),  # sum = 0.90
    pytest.param(valid_record(scores=valid_scores() | {"cs": 0.702}), "scores sum to", id="sum-too-high"),  # sum = 1.002
]


@pytest.mark.parametrize("record, message", BAD_RECORDS)  # runs this test once per case above
def test_broken_record_raises(record, message):
    with pytest.raises(PredictionFileError, match=message):  # passes only if this exact error type is raised
        validate_one_parsed_json(record)


def test_error_message_starts_with_line_number():
    with pytest.raises(PredictionFileError, match=r"^line 7: "):  # ^ = the message must START with "line 7: "
        validate_one_parsed_json(valid_record(label="xyz"), line_no=7)


def test_prediction_file_error_is_a_value_error():
    assert issubclass(PredictionFileError, ValueError)  # so code that catches ValueError also catches it


# ---------------------------------------------------------------------------
# load_predictions: files that MUST load
# ---------------------------------------------------------------------------

def test_load_file_with_scores(tmp_path):
    first = valid_record()  # gold-free prediction for one segment
    second = valid_record(seg_id="Bdb001-cb_0835570_0836020_p1", label="bk", scores=scores_from(bk=1.0))  # a split part
    path = write_jsonl(tmp_path, [json.dumps(first), json.dumps(second)])

    records = load_predictions(path)

    assert records == {  # keyed by seg_id, each value has exactly "label" and "scores"
        first["seg_id"]: {"label": "cs", "scores": first["scores"]},
        second["seg_id"]: {"label": "bk", "scores": second["scores"]},
    }


def test_load_file_with_all_null_scores(tmp_path):
    first = valid_record(scores=None)
    second = valid_record(seg_id="Bdb001-cb_0828718_0829148", label="other", scores=None)
    path = write_jsonl(tmp_path, [json.dumps(first), json.dumps(second)])

    records = load_predictions(str(path))  # also checks that a plain string path works, not only a Path object

    assert records == {
        first["seg_id"]: {"label": "cs", "scores": None},
        second["seg_id"]: {"label": "other", "scores": None},
    }


def test_blank_lines_are_skipped(tmp_path):
    first = valid_record()
    second = valid_record(seg_id="Bdb001-cb_0828718_0829148")
    path = write_jsonl(tmp_path, ["", json.dumps(first), "   ", json.dumps(second), ""])  # empty and spaces-only lines

    assert set(load_predictions(path)) == {first["seg_id"], second["seg_id"]}  # only the 2 real lines are loaded


def test_empty_file_returns_empty_dict(tmp_path):
    # Decision: an empty file loads as {} without an error. It is still caught later, because
    # per_class_metrics raises ValueError when predictions and gold do not have the same seg_ids.
    path = write_jsonl(tmp_path, [])

    assert load_predictions(path) == {}


# ---------------------------------------------------------------------------
# load_predictions: files that MUST fail
# ---------------------------------------------------------------------------

def test_invalid_json_raises_with_line_number(tmp_path):
    path = write_jsonl(tmp_path, [json.dumps(valid_record()), '{"seg_id": "broken"'])  # line 2 is cut off

    with pytest.raises(PredictionFileError, match=r"^line 2: invalid JSON"):
        load_predictions(path)


def test_bad_record_inside_file_reports_its_line(tmp_path):
    lines = [
        json.dumps(valid_record()),
        json.dumps(valid_record(seg_id="Bdb001-cb_0828718_0829148")),
        json.dumps(valid_record(seg_id="Bdb001-c4_0833855_0835255", label="xyz")),  # line 3 is broken
    ]
    path = write_jsonl(tmp_path, lines)

    with pytest.raises(PredictionFileError, match=r"^line 3: "):  # the whole file is rejected and line 3 is named
        load_predictions(path)


def test_line_numbers_count_blank_lines(tmp_path):
    path = write_jsonl(tmp_path, [json.dumps(valid_record()), "", "not json"])  # the bad line is line 3 of the file

    with pytest.raises(PredictionFileError, match=r"^line 3: "):  # the number must match what you see in an editor
        load_predictions(path)


def test_duplicate_seg_id_raises(tmp_path):
    first = valid_record()
    again = valid_record(label="other")  # same seg_id as `first`, different label
    path = write_jsonl(tmp_path, [json.dumps(first), json.dumps(again)])

    with pytest.raises(PredictionFileError, match=r"^line 2: duplicate seg_id"):
        load_predictions(path)


@pytest.mark.parametrize("first_scores, second_scores", [
    pytest.param(valid_scores(), None, id="scores-then-null"),
    pytest.param(None, valid_scores(), id="null-then-scores"),
])
def test_mixed_scores_and_null_raises(tmp_path, first_scores, second_scores):
    first = valid_record(scores=first_scores)
    second = valid_record(seg_id="Bdb001-cb_0828718_0829148", scores=second_scores)
    path = write_jsonl(tmp_path, [json.dumps(first), json.dumps(second)])

    with pytest.raises(PredictionFileError, match=r"^line 2: mixed file"):  # a file is all-scores or all-null
        load_predictions(path)


def test_nan_score_in_file_raises(tmp_path):
    # Decision: NaN must be rejected. Python's json reads the bare word NaN without complaining,
    # so this checks that the range check (not the JSON parser) stops it.
    record = valid_record(scores=scores_from(cs=float("nan"), co=1.0))
    path = write_jsonl(tmp_path, [json.dumps(record)])  # json.dumps writes the score as the bare word NaN

    with pytest.raises(PredictionFileError, match=r"^line 1: score for 'cs' must be a number"):
        load_predictions(path)
