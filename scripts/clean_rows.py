"""Remove non-speech, unlabelled and empty rows from the annotated output.
Reads data/interim/annotated/{rows,links}.jsonl (checked against provenance/c2.json) and
writes the cleaned files to data/interim/cleaned/, a cleaning report and provenance/c2_clean.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotate_rows import recorded, sha256  
from smi.data.annotate.cleaning import (  
    DROP_FLAGS,
    build_cleaning_report,
    check_cleaned,
    clean_links,
    clean_rows,
    is_dropped,
)
from smi.data.annotate.jsonl import write_jsonl
from smi.data.annotate.records import AnnotatedRow, LinkRecord 

TASK_FOLDER = PROJECT_ROOT / "tasks" / "A-data-transformation"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "interim" / "annotated"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "cleaned"
DEFAULT_ANNOTATED_PROVENANCE = TASK_FOLDER / "provenance" / "c2.json"
DEFAULT_REPORT = TASK_FOLDER / "reports" / "cleaning_report.md"
DEFAULT_PROVENANCE = TASK_FOLDER / "provenance" / "c2_clean.json"


def read_jsonl(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def fingerprint_problems(provenance_path: Path, rows_path: Path, links_path: Path) -> list[str]:
    expected = json.loads(provenance_path.read_text(encoding="utf-8"))
    pairs = (("rows", rows_path), ("links", links_path))
    return [
        f"{name} do not match {provenance_path.name}; re-run scripts/annotate_rows.py"
        for name, path in pairs
        if sha256(path) != expected[name]["sha256"]
    ]


def write_provenance(
    path: Path, inputs: tuple[Path, Path], outputs: tuple[Path, Path], counts: dict[str, int]
) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    provenance = {
        "input_rows": recorded(inputs[0], counts["annotated_rows"]),
        "input_links": recorded(inputs[1], counts["annotated_links"]),
        "rows": recorded(outputs[0], counts["cleaned_rows"]),
        "links": recorded(outputs[1], counts["cleaned_links"]),
        "dropped": {flag: counts[flag] for flag in DROP_FLAGS} | {"unique": counts["dropped"]},
        "producing_commit": commit,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="folder with the annotated rows.jsonl and links.jsonl")
    parser.add_argument("--annotated-provenance", type=Path, default=DEFAULT_ANNOTATED_PROVENANCE, help="c2.json to verify the inputs against")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="folder for the cleaned rows.jsonl and links.jsonl")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="cleaning report (Markdown)")
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE, help="provenance file (JSON)")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    rows_in, links_in = args.input_dir / "rows.jsonl", args.input_dir / "links.jsonl"
    problems = fingerprint_problems(args.annotated_provenance, rows_in, links_in)
    if problems:
        print("Inputs rejected; nothing was written:", *problems, sep="\n  ", file=sys.stderr)
        return 1

    annotated = cast(list[AnnotatedRow], read_jsonl(rows_in))
    annotated_links = cast(list[LinkRecord], read_jsonl(links_in))
    cleaned = clean_rows(annotated)
    cleaned_links = clean_links(annotated_links, annotated)

    problems = check_cleaned(annotated, annotated_links, cleaned, cleaned_links)
    if problems:
        print("Contract problems found; nothing was written:", *problems, sep="\n  ", file=sys.stderr)
        return 1

    rows_out, links_out = args.output_dir / "rows.jsonl", args.output_dir / "links.jsonl"
    write_jsonl(cleaned, rows_out)
    write_jsonl(cleaned_links, links_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        build_cleaning_report(annotated, annotated_links, cleaned, cleaned_links),
        encoding="utf-8", newline="\n",
    )
    counts = {
        "annotated_rows": len(annotated), "annotated_links": len(annotated_links),
        "cleaned_rows": len(cleaned), "cleaned_links": len(cleaned_links),
        "dropped": sum(is_dropped(row) for row in annotated),
        **{flag: sum(row[flag] for row in annotated) for flag in DROP_FLAGS},
    }
    write_provenance(args.provenance, (rows_in, links_in), (rows_out, links_out), counts)
    print(f"Wrote {len(cleaned)} cleaned rows and {len(cleaned_links)} links to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
