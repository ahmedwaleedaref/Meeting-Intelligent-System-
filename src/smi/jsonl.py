"""Shared JSONL reading: one JSON value per line, used by the release script, load_gold and load_predictions."""

import json
from collections.abc import Iterable, Iterator


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    # json.loads keeps the last value of a repeated key without a word ; refuse it instead
    obj = dict(pairs)
    if len(obj) != len(pairs):
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise _DuplicateKeyError(key)
            seen.add(key)
    return obj


def iter_jsonl(lines: Iterable[str], where: str = "", error: type[ValueError] = ValueError) -> Iterator[tuple[int, object]]:
    """
    yield (line_no, parsed value) for every non-blank line ; line numbers count blank lines too.
    where : prefix of every error message (e.g. a file name)
    error : exception class raised on invalid JSON or a key repeated inside one object
    """
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except _DuplicateKeyError as e:
            raise error(f"{where}line {line_no}: duplicate key {e.args[0]!r}") from None
        except json.JSONDecodeError as e:
            raise error(f"{where}line {line_no}: invalid JSON ({e.msg})") from e
        yield line_no, value
