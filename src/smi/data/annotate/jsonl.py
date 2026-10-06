"""Read and write JSONL maintaining existing work."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from smi.data.annotate.records import ParsedRow


def read_parsed_rows(path: Path) -> list[ParsedRow]:
    with path.open(encoding="utf-8") as file:
        return [cast(ParsedRow, json.loads(line)) for line in file]


def write_jsonl(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            file.write("\n")
