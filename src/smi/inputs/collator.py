from typing import TypedDict

import torch
from transformers import PreTrainedTokenizerBase

from smi.inputs.dataset import SegmentItem


class CollatorBatch(TypedDict):
    """
    one batch , ready for the model ; B = number of items , L = longest input in this batch.
    input_ids      : [B, L] token ids , padded to L
    attention_mask : [B, L] 1 = real token , 0 = pad
    labels         : [B] index into LABELS ; -100 for items with label None (test)
    seg_ids        : the B target seg_ids , in batch order (a list : tensors cannot hold strings)
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    seg_ids: list[str]


class SegmentCollator:
    """
    turns a list of SegmentItem (one batch from the sampler) into one CollatorBatch.
    model-dependent only through the tokenizer it is given ; no family branches :
    a decoder later = the same class with a suffix and a tokenizer whose pad token is set.
    rule : items are read-only , never modified here (they are stored in the dataset).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        separator: str,
        markers: tuple[str, str] = ("[T]", "[/T]"),
        suffix: str = "",
    ) -> None:
        """
        tokenizer  : the model's tokenizer , loaded from the same checkpoint name as the model ;
                     its pad token and padding side are used as given
        max_length : maximum number of tokens per input , special tokens included
        separator  : text between two segments
        markers    : (open , close) text wrapped around the target ; spaces , if wanted , are part of them
        suffix     : text added after the last segment ; "" for the encoder

        counters , since this collator was created (read them after an epoch to see how often truncation fires) :
        n_items     : number of items collated
        n_truncated : number of items whose input was longer than max_length and got cut
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator = separator
        self.markers = markers
        self.suffix = suffix
        self.n_items = 0
        self.n_truncated = 0

    def render(self, item: SegmentItem) -> str:
        """
        the input text of one item :
        separator.join(reversed(left) + [open + target_text + close] + right) + suffix
        left is closest-first in the item , so it is reversed here to get speaking order.
        """
        open_marker = self.markers[0]
        close_marker = self.markers[1]

        segments = []
        # left is closest-first : walk it backwards to get speaking order
        for text in reversed(item["left"]):
            segments.append(text)
        segments.append(open_marker + item["target_text"] + close_marker)
        for text in item["right"]:
            segments.append(text)

        # the suffix is glued to the end , not joined with the separator
        return self.separator.join(segments) + self.suffix

    def __call__(self, items: list[SegmentItem]) -> CollatorBatch:
        """
        render every item , then one tokenizer call for the whole batch :
        special tokens added by the tokenizer , padded to the longest input in the batch ,
        truncated to max_length by the tokenizer (cuts from the end ; baseline rule , revisit for k > 3) ;
        token_type_ids are dropped.
        labels : one per item (sequence classification) ; -100 for label None.
        updates n_items and n_truncated.
        """
        texts = []
        for item in items:
            texts.append(self.render(item))

        encoded = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        # an input can only have been cut if it now fills max_length exactly ;
        # for those few , tokenize again without truncation to see whether it was longer
        lengths = encoded["attention_mask"].sum(dim=1)
        for i in range(len(texts)):
            if lengths[i] == self.max_length:
                full_ids = self.tokenizer(texts[i])["input_ids"]
                if len(full_ids) > self.max_length:
                    self.n_truncated = self.n_truncated + 1
        self.n_items = self.n_items + len(texts)

        labels = []
        seg_ids = []
        for item in items:
            if item["label"] is None:
                labels.append(-100)
            else:
                labels.append(item["label"])
            seg_ids.append(item["seg_id"])

        return CollatorBatch(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=torch.tensor(labels, dtype=torch.long),
            seg_ids=seg_ids,
        )
