"""Hand-built C2 inputs and C3 releases for the Task 3 tests."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.make_release import make_release
from smi.data.release import load_release
from smi.data.splits import build_split_provenance

SOURCE_ARCHIVE = {"archive_name": "fixture.zip", "release": "fixture-1", "sha256": "b" * 64}
DECISIONS = {f"D{number}": f"fixture decision {number}" for number in range(1, 17)}


def formatter_row(
    seg_id: str,
    meeting: str,
    position: int,
    speaker: str,
    text: str,
) -> dict[str, Any]:
    return {
        "seg_id": seg_id,
        "meeting": meeting,
        "position": position,
        "speaker_id": speaker,
        "text": text,
        "source_da_id": f"SOURCE-{seg_id}",
        "label": "SECRET_LABEL",
        "target_tags": ["SECRET_TAG"],
        "links": ["SECRET_LINK"],
        "split": "SECRET_SPLIT",
        "fold": 4,
    }


def make_row(
    seg_id: str,
    meeting: str,
    position: int,
    *,
    target_tags: Sequence[str] = (),
    label: str | None = None,
    speaker: str = "speaker-1",
    text: str = "text",
) -> dict[str, Any]:
    tags = list(target_tags)
    proposal = "cs" in tags or "co" in tags
    chosen = label or (tags[0] if tags else "other")
    return {
        "seg_id": seg_id,
        "meeting": meeting,
        "agent": "A",
        "channel": "c1",
        "speaker_id": speaker,
        "src_start": float(position),
        "src_end": float(position + 1),
        "start": float(position),
        "end": float(position + 1),
        "timing_method": "element",
        "position": position,
        "source_da_id": f"{meeting}.A.dialogueact{position}",
        "element_ordinal": position,
        "part_index": 0,
        "n_parts": 1,
        "raw_type": "s",
        "original_type": "s",
        "part_type": "s",
        "adjacency_raw": None,
        "comment": None,
        "text": text,
        "words": [],
        "quality_flags": [],
        "label": chosen,
        "label_bk_merged": chosen,
        "target_tags": tags,
        "is_multi_target": len(tags) >= 2,
        "is_layer2_proposal": proposal,
        "general_tags": [],
        "disruption_markers": [],
    }


def link(proposal_id: str, chains: list[list[str]], status: str | None = None) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "responder_chains": chains,
        "has_response": bool(chains),
        "status": status or ("ok" if chains else "no_pair"),
        "adjacency_tokens": ["1a"] if chains else [],
        "pair_group": f"pair-{proposal_id}" if chains else None,
    }


def c2_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return C2 rows, links and split provenance for three meetings (train, val, test)."""
    rows = [
        make_row("m1-p", "m1", 0, target_tags=["cs"]),
        make_row("m1-r", "m1", 1, speaker="speaker-2"),
        make_row("m1-q", "m1", 2, target_tags=["cs", "cc"], label="cc"),
        make_row("m2-p", "m2", 0, target_tags=["co"]),
        make_row("m2-r", "m2", 1, speaker="speaker-2"),
        make_row("m3-a", "m3", 0),
    ]
    links = [link("m1-p", [["m1-r"]]), link("m1-q", []), link("m2-p", [["m2-r"]])]
    provenance = build_split_provenance(
        ["m1", "m2", "m3"],
        {"train": ["m1"], "val": ["m2"], "test": ["m3"]},
        split_source="fixture split lists",
    )
    return rows, links, provenance


def write_jsonl_lines(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def write_inputs(
    root: Path,
    rows: list[dict[str, Any]] | None = None,
    links: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Path]:
    default_rows, default_links, default_provenance = c2_fixture()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": root / "c2_rows.jsonl",
        "links": root / "c2_links.jsonl",
        "splits": root / "split_v1.json",
        "source": root / "source_archive.json",
        "decisions": root / "decisions.json",
    }
    write_jsonl_lines(paths["rows"], rows or default_rows)
    write_jsonl_lines(paths["links"], links or default_links)
    paths["splits"].write_text(json.dumps(provenance or default_provenance) + "\n", encoding="utf-8")
    paths["source"].write_text(json.dumps(SOURCE_ARCHIVE) + "\n", encoding="utf-8")
    paths["decisions"].write_text(json.dumps(DECISIONS) + "\n", encoding="utf-8")
    return paths


def build_release(
    root: Path,
    name: str = "release",
    *,
    inputs: dict[str, Path] | None = None,
    **options: Any,
) -> tuple[Path, str, dict[str, Path]]:
    paths = inputs or write_inputs(root / "inputs")
    output = root / name
    fingerprint = make_release(
        paths["rows"], paths["links"], paths["splits"], paths["source"], paths["decisions"], output, **options
    )
    return output, fingerprint, paths


def read_release_parts(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [json.loads(line) for line in (output / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    links = [json.loads(line) for line in (output / "links.jsonl").read_text(encoding="utf-8").splitlines()]
    return rows, links, json.loads((output / "splits.json").read_text(encoding="utf-8"))


def rewrite_checksums(output: Path) -> None:
    lines = []
    for name in ("links.jsonl", "metadata.json", "rows.jsonl", "splits.json"):
        lines.append(f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}")
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_built(root: Path) -> Any:
    output, _, _ = build_release(root)
    return load_release(output)
