from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from smi.inputs.dataset import SegmentDataset


class LengthBucketBatchSampler(Sampler[list[int]]):
    """
    groups dataset indices into batches of similar length , so padding stays small.
    length of an item = number of characters of left + target_text + right (model-independent).
    a batch is one unit on one GPU ; splitting batches across GPUs is left to Accelerate.

    train (shuffle=True) : shuffle all indices -> cut into chunks of chunk_batches batches ->
                           sort by length inside each chunk -> cut into batches -> shuffle the batch order ;
                           done once in __init__ : every epoch yields the same batches in the same order.
                           chunks keep batches mixed : fully sorting would put all short items
                           (mostly backchannels) together and make batches nearly single-class.
    eval (shuffle=False) : all indices sorted by length , longest first (an out-of-memory error shows up
                           in the first batch) , same order every time.
    the last batch may be smaller than batch_size ; it is kept , so every item is used.
    """

    def __init__(
        self,
        dataset: SegmentDataset,
        batch_size: int,
        chunk_batches: int,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        """
        dataset       : the items to batch ; lengths are computed once here , not every epoch
        batch_size    : number of items per batch
        chunk_batches : number of batches per chunk (train only)
        shuffle       : True for train , False for eval
        raises ValueError on : batch_size or chunk_batches < 1
        seed          : seeds the sampler's own torch.Generator , used once here (train only) ;
                        independent of the global random state , so every model gets the same batches
                        with the same seed
        """
        if batch_size < 1 or chunk_batches < 1:
            raise ValueError(f"batch_size and chunk_batches must be >= 1, got {batch_size} and {chunk_batches}")

        generator = torch.Generator()
        generator.manual_seed(seed)

        # length of every item, computed once : characters of target + left + right
        lengths = []
        for i in range(len(dataset)):
            item = dataset[i]
            length = len(item["target_text"])
            for text in item["left"]:
                length = length + len(text)
            for text in item["right"]:
                length = length + len(text)
            lengths.append(length)

        batches = []
        if shuffle:
            # train : shuffle -> chunks -> sort inside each chunk -> batches -> shuffle the batch order
            indices = torch.randperm(len(dataset), generator=generator).tolist()

            chunk_size = chunk_batches * batch_size
            for start in range(0, len(indices), chunk_size):
                chunk = indices[start:start + chunk_size]
                chunk = sorted(chunk, key=lambda i: lengths[i])
                for batch_start in range(0, len(chunk), batch_size):
                    batches.append(chunk[batch_start:batch_start + batch_size])

            batch_order = torch.randperm(len(batches), generator=generator).tolist()
            shuffled_batches = []
            for b in batch_order:
                shuffled_batches.append(batches[b])
            batches = shuffled_batches
        else:
            # eval : all indices sorted by length, longest first -> batches
            indices = sorted(range(len(dataset)), key=lambda i: lengths[i], reverse=True)
            for batch_start in range(0, len(indices), batch_size):
                batches.append(indices[batch_start:batch_start + batch_size])

        self.batches = batches

    def __iter__(self) -> Iterator[list[int]]:
        """yield the batches built in __init__ , each a list of dataset indices ; the same every epoch."""
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        """number of batches per epoch = ceil(len(dataset) / batch_size) ; must match what __iter__ yields
        (the LR scheduler uses it to count steps)."""
        return len(self.batches)
