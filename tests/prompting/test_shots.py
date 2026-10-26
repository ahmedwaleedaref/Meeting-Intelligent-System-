"""tests for smi.prompting.shots : picking few-shot examples from train."""

import json

import pytest

from smi.labels import LABELS
from smi.prompting.shots import pick_shots


def write_split(folder, n_meetings=2, per_label=3):
    """a small split folder : each meeting has per_label segments of every class , interleaved."""
    folder.mkdir(parents=True)
    for m in range(n_meetings):
        rows = []
        for i in range(per_label * len(LABELS)):
            rows.append({"seg_id": f"M{m}-{i}", "position": i, "speaker": "a", "text": f"text {m}-{i}", "label": LABELS[i % len(LABELS)]})
        with open(folder / f"M{m}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return folder


@pytest.fixture
def train_dir(tmp_path):
    return write_split(tmp_path / "train")


def test_one_example_per_class_in_label_order(train_dir):
    shots = pick_shots(train_dir, past_k=0, future_k=0, per_class=1, seed=0)
    assert [LABELS[shot["label"]] for shot in shots] == list(LABELS)


def test_per_class_two_gives_two_different_segments_of_each_class(train_dir):
    shots = pick_shots(train_dir, past_k=0, future_k=0, per_class=2, seed=0)
    assert [LABELS[shot["label"]] for shot in shots] == [label for label in LABELS for _ in range(2)]
    assert len({shot["seg_id"] for shot in shots}) == 14


def test_same_seed_same_examples(train_dir):
    first = pick_shots(train_dir, past_k=0, future_k=0, per_class=1, seed=7)
    second = pick_shots(train_dir, past_k=0, future_k=0, per_class=1, seed=7)
    assert [s["seg_id"] for s in first] == [s["seg_id"] for s in second]


def test_same_segments_for_every_window_size_and_context_is_included(train_dir):
    no_context = pick_shots(train_dir, past_k=0, future_k=0, per_class=1, seed=3)
    with_context = pick_shots(train_dir, past_k=1, future_k=1, per_class=1, seed=3)
    assert [s["seg_id"] for s in no_context] == [s["seg_id"] for s in with_context]
    # an interleaved meeting gives every segment a neighbour on at least one side
    assert all(len(s["left"]) + len(s["right"]) > 0 for s in with_context)


def test_refuses_a_folder_that_is_not_train(tmp_path):
    val_dir = write_split(tmp_path / "val")
    with pytest.raises(ValueError, match="train split"):
        pick_shots(val_dir, past_k=0, future_k=0, per_class=1, seed=0)
