"""tests for smi.prompting.retrieve : each segment's own few-shot examples."""

import json
from types import SimpleNamespace

import pytest
import torch

from smi.inputs.dataset import SegmentDataset
from smi.labels import LABELS
from smi.prompting.retrieve import (embed, example_pool, nearest_per_class, random_per_class, read_shots_file,
                                    write_shots_file)
from tests.prompting.test_shots import write_split

TWO_PER_CLASS = [c for c in range(len(LABELS)) for _ in range(2)]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"))]


def unit(rows):
    return torch.nn.functional.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def toward_query(far_first: bool):
    """two pool rows per class , one pointing more toward the query direction (the last axis) than the other."""
    rows = []
    for c in range(len(LABELS)):
        far, close = [0.0] * 8, [0.0] * 8
        far[c], far[7], close[c], close[7] = 1.0, 0.5, 1.0, 2.0
        rows += [far, close] if far_first else [close, far]
    return unit(rows)


@pytest.mark.parametrize("device", DEVICES)
def test_nearest_per_class_gives_each_class_closest_first_in_label_order(device):
    query, pool = unit([[0.0] * 7 + [1.0]]).to(device), toward_query(far_first=True).to(device)
    rows, sims = nearest_per_class(query, pool, TWO_PER_CLASS, per_class=2)
    assert rows == [[r for c in range(len(LABELS)) for r in (2 * c + 1, 2 * c)]]
    assert all(sims[0][2 * c] > sims[0][2 * c + 1] for c in range(len(LABELS)))


def test_nearest_per_class_is_the_same_in_chunks():
    queries = unit(torch.randn(5, 8, generator=torch.Generator().manual_seed(0)).tolist())
    pool = toward_query(far_first=False)
    (rows, sims), (chunked_rows, chunked_sims) = (nearest_per_class(queries, pool, TWO_PER_CLASS, 2, chunk=chunk)
                                                  for chunk in (1024, 2))
    assert chunked_rows == rows
    assert all(chunked == pytest.approx(full) for chunked, full in zip(chunked_sims, sims))


def test_nearest_per_class_needs_enough_examples_of_every_class():
    with pytest.raises(ValueError, match="fewer than 3"):
        nearest_per_class(unit([[0.0] * 7 + [1.0]]), toward_query(far_first=True), TWO_PER_CLASS, per_class=3)


def test_random_per_class_draws_every_class_and_repeats_with_its_seed():
    draws = random_per_class(20, TWO_PER_CLASS * 3, per_class=2, seed=0)
    assert all([TWO_PER_CLASS[row % 14] for row in rows] == TWO_PER_CLASS for rows in draws)
    assert draws == random_per_class(20, TWO_PER_CLASS * 3, per_class=2, seed=0)
    assert len({tuple(rows) for rows in draws}) > 1


def test_example_pool_keeps_one_example_per_class_and_transcript(tmp_path):
    texts_labels = [("yeah", "bk"), ("yeah", "bk"), ("yeah", "aa"), ("OK", "bk"), ("yeah", "bk")]
    (tmp_path / "train").mkdir()
    with open(tmp_path / "train" / "M0.jsonl", "w", encoding="utf-8") as f:
        for i, (text, label) in enumerate(texts_labels):
            f.write(json.dumps({"seg_id": f"M0-{i}", "position": i, "speaker": "a", "text": text, "label": label}) + "\n")
    assert example_pool(SegmentDataset(tmp_path / "train", past_k=0, future_k=0)) == [0, 2, 3]


class FakeEncoder(torch.nn.Module):
    """last layer = (token id , 1) at every position."""

    def __init__(self):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, output_hidden_states):
        hidden = torch.stack([input_ids.float(), torch.ones(input_ids.shape)], dim=-1)
        return SimpleNamespace(hidden_states=(hidden,))


def test_embed_pools_the_first_token_or_the_mean_of_the_real_tokens():
    batches = [{"input_ids": torch.tensor([[3, 5, 0]]), "attention_mask": torch.tensor([[1, 1, 0]]), "seg_ids": ["a"]}]
    assert embed(FakeEncoder(), batches, "cls")["a"].tolist() == pytest.approx(unit([[3.0, 1.0]])[0].tolist())
    assert embed(FakeEncoder(), batches, "mean")["a"].tolist() == pytest.approx(unit([[4.0, 1.0]])[0].tolist())


def test_a_shots_file_reads_back_as_train_items_for_its_own_data_and_window(tmp_path):
    train_dir = write_split(tmp_path / "train", n_meetings=1, per_label=1)
    settings = {"data_version": "toy", "past_k": 0, "future_k": 0}
    write_shots_file(tmp_path / "shots", settings, {"v-0": ["M0-3", "M0-0"]}, None)
    shots, read_settings, sha256 = read_shots_file(tmp_path / "shots", train_dir, 0, 0, "toy")
    assert [item["seg_id"] for item in shots["v-0"]] == ["M0-3", "M0-0"] and read_settings == settings
    with pytest.raises(ValueError, match="past_k"):
        read_shots_file(tmp_path / "shots", train_dir, 1, 0, "toy")
    with pytest.raises(ValueError, match="data"):
        read_shots_file(tmp_path / "shots", train_dir, 0, 0, "v3")
