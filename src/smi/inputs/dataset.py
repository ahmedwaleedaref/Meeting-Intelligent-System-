from pathlib import Path
from typing import TypedDict
from torch.utils.data import Dataset

from smi.jsonl import iter_jsonl
from smi.labels import LABELS


class SegmentItem(TypedDict):
    """
    one item = one target segment with its context window, shared by every model family.
    seg_id      : id of the target segment (join key with the harness)
    label       : index of the gold label in LABELS ; None for the test split
    left        : texts of the segments before the target ; index 0 = closest , index k-1 = farthest
                  (reverse speaking order : renderers must reverse it before joining)
    target_text : text of the target segment
    right       : texts of the segments after the target ; index 0 = closest , index k-1 = farthest
    """
    seg_id: str
    label: int | None
    left: list[str]
    target_text: str
    right: list[str]


class SegmentDataset(Dataset):
    """
    every segment of one split of the per-meeting files (scripts/make_meeting_files.py),
    each with up to past_k segments before and future_k after it, clipped at meeting boundaries.
    no tokenizer and no rendering : items are plain text , so encoder , decoder and prompting share it.
    """

    def __init__(self, split_dir: str | Path, past_k: int, future_k: int, allow_test: bool = False) -> None:
        """
        split_dir  : a split folder (e.g. data/v4_meetings/val) ; every *.jsonl in it is one meeting
        past_k     : number of segments before the target (0 = none)
        future_k   : number of segments after the target (0 = none)
        allow_test : the test split is read only when True , and its items then have label None (PROTOCOL §7)

        meetings are read in file-name order and segments in position order,
        so index i is the same segment on every machine.
        raises ValueError on : the test split without allow_test=True ,
                               a folder with no *.jsonl files ,
                               negative past_k / future_k ,
                               a label not in LABELS
        """
        if past_k < 0 or future_k < 0:
            raise ValueError(f"past_k and future_k must be >= 0, got {past_k} and {future_k}")

        split_dir = Path(split_dir)
        # resolve() so that "test/.." tricks and symlinks are judged by the real folder, like load_gold
        is_test = split_dir.resolve().name == "test"
        if is_test and not allow_test:
            raise ValueError(f"{split_dir}: test split is read only with allow_test=True (PROTOCOL §7)")

        files = sorted(split_dir.glob("*.jsonl"))
        if len(files) == 0:
            raise ValueError(f"{split_dir}: no *.jsonl files in folder")

        data_list = []
        for file in files:
            # one file = one meeting ; its rows are already in position order (make_meeting_files.py)
            rows = []
            with open(file, "r", encoding="utf-8") as f:
                for line_no, row in iter_jsonl(f, where=f"{file.name} "):
                    rows.append(row)

            for i in range(len(rows)):
                row = rows[i]

                # previous segments, closest first, stop at the start of the meeting
                left = []
                for step in range(1, past_k + 1):
                    j = i - step
                    if j < 0:
                        break
                    left.append(rows[j]["text"])

                # next segments, closest first, stop at the end of the meeting
                right = []
                for step in range(1, future_k + 1):
                    j = i + step
                    if j >= len(rows):
                        break
                    right.append(rows[j]["text"])

                if is_test:
                    label = None
                else:
                    if row["label"] not in LABELS:
                        raise ValueError(f"{file.name}: seg_id {row['seg_id']!r} has label {row['label']!r}, not in LABELS")
                    label = LABELS.index(row["label"])

                item = SegmentItem(
                    seg_id=row["seg_id"],
                    label=label,
                    left=left,
                    target_text=row["text"],
                    right=right,
                )
                data_list.append(item)

        self.data = data_list


    def __len__(self) -> int:
        """number of segments in the split."""
        return len(self.data)

    def __getitem__(self, index: int) -> SegmentItem:
        """the index-th segment (meetings in file-name order , segments in position order) with its context."""
        return self.data[index]
