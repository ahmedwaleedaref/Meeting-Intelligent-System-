"""
each segment's own few-shot examples : per class , the train examples most similar to it (KATE , Liu et al. 2021 ,
"What Makes Good In-Context Examples for GPT-3?") , or random ones (the control).
similarity = cosine between an encoder's embeddings of the segments (scripts/retrieve_shots.py makes them).

the result is a shots file , a folder with :
    settings.json  how the examples were chosen (encoder , pooling , per_class , data version , context window)
    shots.jsonl    one line per segment : {"seg_id" , "shots": train seg_ids in prompt order , "similarity"}
scripts/trial_prompting.py --shots-file reads it ; the prompt order is LABELS order (per_class examples of cs first ,
... , other last) , as pick_shots , so only the choice of examples changes.
"""

import hashlib
import json
import random
from pathlib import Path

import torch

from smi.inputs.dataset import SegmentDataset, SegmentItem
from smi.labels import LABELS
from smi.prompting.render import render_transcript

POOLINGS: tuple[str, ...] = ("cls", "mean")


def example_pool(train: SegmentDataset) -> list[int]:
    """
    the train items examples are chosen from : one per (class , transcript) , the first in dataset order.
    identical transcripts would show the model the same example twice ; at k = 0 hundreds of segments are just
    "yeah" or "OK" , and the copies would also tie in similarity.
    returns their indices in train , in dataset order
    """
    seen = set()
    pool = []
    for index in range(len(train)):
        item = train[index]
        key = (item["label"], render_transcript(item))
        if key not in seen:
            seen.add(key)
            pool.append(index)
    return pool


def embed(model, dataloader, pooling: str, fp16: bool = False) -> dict[str, torch.Tensor]:
    """
    seg_id -> the L2-normalised embedding (fp32 , on the CPU) of every item of dataloader (smi.inputs.loader).
    pooling : "cls"  = the last layer's first token (the token a fine-tuned classifier reads)
              "mean" = the mean of the last layer over the real tokens (padding left out)
    fp16    : run the forward pass under fp16 autocast (GPU only)
    """
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}, expected one of {POOLINGS}")
    model.eval()
    device = next(model.parameters()).device
    vectors = {}
    with torch.no_grad():
        for batch in dataloader:
            mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                output = model(input_ids=batch["input_ids"].to(device), attention_mask=mask, output_hidden_states=True)
            hidden = output.hidden_states[-1].float()
            if pooling == "cls":
                pooled = hidden[:, 0]
            else:
                weights = mask.unsqueeze(-1).float()
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
            for seg_id, vector in zip(batch["seg_ids"], torch.nn.functional.normalize(pooled, dim=-1).cpu()):
                vectors[seg_id] = vector
    return vectors


def nearest_per_class(queries: torch.Tensor, pool: torch.Tensor, pool_labels: list[int], per_class: int,
                      chunk: int = 1024) -> tuple[list[list[int]], list[list[float]]]:
    """
    for each query , the per_class most similar pool rows of every class.
    queries , pool : [n, d] and [m, d] , every row L2-normalised (a dot product is then the cosine)
    pool_labels    : the class index (in LABELS) of each pool row
    returns (rows , similarities) : per query , 7 × per_class pool rows in LABELS order , the most similar first
            within a class , and their cosine similarities
    raises ValueError if a class has fewer than per_class rows in the pool
    """
    labels = torch.tensor(pool_labels, device=pool.device)  # on the pool's device : topk's positions index these rows
    by_class = [torch.nonzero(labels == c).flatten() for c in range(len(LABELS))]
    for label, rows in zip(LABELS, by_class):
        if len(rows) < per_class:
            raise ValueError(f"class {label!r} has {len(rows)} examples in the pool, fewer than {per_class}")

    picked: list[list[int]] = [[] for _ in range(len(queries))]
    similarity: list[list[float]] = [[] for _ in range(len(queries))]
    for start in range(0, len(queries), chunk):
        block = queries[start : start + chunk] @ pool.T  # [chunk, m] cosines
        for rows in by_class:  # in LABELS order , so each query's list comes out class by class
            values, positions = block[:, rows].topk(per_class, dim=1)
            for i, (chosen, cosines) in enumerate(zip(rows[positions].tolist(), values.tolist())):
                picked[start + i].extend(chosen)
                similarity[start + i].extend(cosines)
    return picked, similarity


def random_per_class(n_queries: int, pool_labels: list[int], per_class: int, seed: int) -> list[list[int]]:
    """
    the control : for each query , per_class different pool rows of every class drawn at random , in LABELS order.
    one private generator , so the same seed always gives the same examples
    raises ValueError if a class has fewer than per_class rows in the pool
    """
    by_class = [[row for row, label in enumerate(pool_labels) if label == c] for c in range(len(LABELS))]
    for label, rows in zip(LABELS, by_class):
        if len(rows) < per_class:
            raise ValueError(f"class {label!r} has {len(rows)} examples in the pool, fewer than {per_class}")
    rng = random.Random(seed)
    return [[row for rows in by_class for row in rng.sample(rows, per_class)] for _ in range(n_queries)]


def write_shots_file(out: Path, settings: dict, shots: dict[str, list[str]],
                     similarity: dict[str, list[float]] | None) -> None:
    """write the shots file (module docstring) into out ; fails if its files exist : outputs are never overwritten."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "settings.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(settings, indent=2) + "\n")
    with open(out / "shots.jsonl", "x", encoding="utf-8") as f:
        for seg_id, train_ids in shots.items():
            row = {"seg_id": seg_id, "shots": train_ids, "similarity": None if similarity is None else similarity[seg_id]}
            f.write(json.dumps(row) + "\n")


def read_shots_file(folder: Path, train_dir: Path, past_k: int, future_k: int,
                    data_version: str) -> tuple[dict[str, list[SegmentItem]], dict, str]:
    """
    a shots file , with its examples as train items rendered with this context window.
    returns (seg_id -> its examples in prompt order , the file's settings , sha256 of shots.jsonl)
    raises ValueError if the file was made for other data or another context window ,
    or names a segment that is not in train_dir
    """
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    if settings["data_version"] != data_version:
        raise ValueError(f"{folder}: made from data {settings['data_version']}, the run uses {data_version}")
    if (settings["past_k"], settings["future_k"]) != (past_k, future_k):
        raise ValueError(f"{folder}: made for past_k {settings['past_k']} , future_k {settings['future_k']} , "
                         f"the run uses {past_k} , {future_k}")
    if Path(train_dir).resolve().name != "train":
        raise ValueError(f"{train_dir}: few-shot examples must come from the train split")

    train = SegmentDataset(train_dir, past_k=past_k, future_k=future_k)
    by_id = {train[i]["seg_id"]: train[i] for i in range(len(train))}
    raw = (folder / "shots.jsonl").read_bytes()
    shots = {}
    for row in map(json.loads, raw.decode("utf-8").splitlines()):
        missing = [seg_id for seg_id in row["shots"] if seg_id not in by_id]
        if missing:
            raise ValueError(f"{folder}: {row['seg_id']}'s examples {missing} are not in {train_dir}")
        shots[row["seg_id"]] = [by_id[seg_id] for seg_id in row["shots"]]
    return shots, settings, hashlib.sha256(raw).hexdigest()
