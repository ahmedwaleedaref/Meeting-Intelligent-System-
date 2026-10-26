"""
pick the solved examples shown in few-shot prompts : a fixed number per class , drawn at random
from the train split with a fixed seed , so every run with the same settings shows the same examples.
"""

import random
from pathlib import Path

from smi.inputs.dataset import SegmentDataset, SegmentItem
from smi.labels import LABELS


def pick_shots(train_dir: str | Path, past_k: int, future_k: int, per_class: int, seed: int) -> list[SegmentItem]:
    """
    per_class examples of each class , in LABELS order (all cs examples first , ... , other last).
    train_dir : the train split folder , e.g. data/v3_meetings/train
    past_k , future_k : the same context window as the segments being classified , so the examples
                        look exactly like the question
    seed : the same seed always picks the same segments , for every k (the Dataset has one item per segment ,
           in the same order , whatever the window size)
    raises ValueError if train_dir is not a train folder : examples from val or test would leak answers
    """
    if Path(train_dir).resolve().name != "train":
        raise ValueError(f"{train_dir}: few-shot examples must come from the train split")

    dataset = SegmentDataset(train_dir, past_k=past_k, future_k=future_k)

    # positions of the items of each class : by_class[0] = indices of every "cs" segment , etc.
    by_class: list[list[int]] = [[] for _ in LABELS]
    for index in range(len(dataset)):
        by_class[dataset[index]["label"]].append(index)

    # one private random generator : the result does not depend on any other random code in the program
    rng = random.Random(seed)
    shots = []
    for class_indices in by_class:
        for index in rng.sample(class_indices, per_class):  # per_class different segments of this class
            shots.append(dataset[index])
    return shots
