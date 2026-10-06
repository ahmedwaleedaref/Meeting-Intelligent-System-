"""Annotate the parsed rows and proposal links."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smi.data.annotate.contract import check_contract  
from smi.data.annotate.jsonl import read_parsed_rows, write_jsonl  
from smi.data.annotate.links.groups import build_groups  
from smi.data.annotate.links.proposals import build_links
from smi.data.annotate.records import AnnotatedRow, LinkRecord, ParsedRow
from smi.data.annotate.report import build_report
from smi.data.annotate.rows import annotate_row  

TASK_FOLDER = PROJECT_ROOT / "tasks" / "A-data-transformation"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "parsed" / "rows.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "annotated"
DEFAULT_REPORT = TASK_FOLDER / "reports" / "annotation_report.md"
DEFAULT_PROVENANCE = TASK_FOLDER / "provenance" / "c2.json"


def annotate_corpus(
    parsed: Sequence[ParsedRow],
) -> tuple[list[AnnotatedRow], list[LinkRecord], Counter[str]]:
    annotated = [annotate_row(row) for row in parsed]
    links: list[LinkRecord] = []
    notes: Counter[str] = Counter()
    for _, meeting_rows in groupby(annotated, key=lambda row: row["meeting"]):
        rows = list(meeting_rows)
        grouping = build_groups(rows)
        notes.update(grouping.notes)
        links.extend(build_links(rows, grouping))
    return annotated, links, notes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded(path: Path, row_count: int) -> dict[str, object]:
    relative = Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()
    return {"path": relative, "sha256": sha256(path), "row_count": row_count}


def write_provenance(
    provenance_path: Path, input_path: Path, rows_path: Path, links_path: Path, counts: tuple[int, int, int]
) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    parsed_count, annotated_count, link_count = counts
    provenance = {
        "input": recorded(input_path, parsed_count),
        "rows": recorded(rows_path, annotated_count),
        "links": recorded(links_path, link_count),
        "producing_commit": commit,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="C1 rows.jsonl to annotate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="folder for rows.jsonl and links.jsonl")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="annotation report (Markdown)")
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE, help="provenance file (JSON)")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    parsed = read_parsed_rows(args.input)
    annotated, links, notes = annotate_corpus(parsed)

    problems = check_contract(parsed, annotated, links)
    if problems:
        print("Contract problems found; nothing was written:", *problems, sep="\n  ", file=sys.stderr)
        return 1

    rows_path = args.output_dir / "rows.jsonl"
    links_path = args.output_dir / "links.jsonl"
    write_jsonl(annotated, rows_path)
    write_jsonl(links, links_path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(annotated, links, notes), encoding="utf-8", newline="\n")
    counts = (len(parsed), len(annotated), len(links))
    write_provenance(args.provenance, args.input, rows_path, links_path, counts)
    print(f"Wrote {len(annotated)} annotated rows and {len(links)} links to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
