from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from smi.inputs.collator import SegmentCollator
from smi.inputs.dataset import SegmentDataset
from smi.inputs.sampler import LengthBucketBatchSampler


def build_dataloader(
    split_dir: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    past_k: int,
    future_k: int,
    batch_size: int,
    chunk_batches: int,
    shuffle: bool,
    max_length: int,
    separator: str,
    markers: tuple[str, str] = ("[T]", "[/T]"),
    suffix: str = "",
    seed: int = 0,
    allow_test: bool = False,
) -> DataLoader:
    """
    the input pipeline of one split : SegmentDataset -> LengthBucketBatchSampler -> SegmentCollator -> DataLoader.
    every argument is passed through unchanged to the object that owns it (see their docstrings).

    num_workers = 0 : batches are prepared in the main process ; loading costs a few ms per batch ,
                      and the collator's counters (n_items , n_truncated) only count in the main process.
    pin_memory      : on when a GPU is available (faster copies to the GPU).
    the collator is reachable as dataloader.collate_fn , e.g. to read n_truncated after an epoch.
    """
    dataset = SegmentDataset(split_dir, past_k, future_k, allow_test=allow_test)
    sampler = LengthBucketBatchSampler(dataset, batch_size, chunk_batches, shuffle, seed=seed)
    collator = SegmentCollator(tokenizer, max_length, separator, markers=markers, suffix=suffix)

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
