"""Build the deterministic C1 parsed-row artifact from the local NXT release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "icsi_core_nxt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "parsed" / "rows.jsonl"
REPORT_PATH = (
    PROJECT_ROOT
    / "tasks"
    / "A-data-transformation"
    / "reports"
    / "parse_validation.md"
)
PROVENANCE_DIRECTORY = PROJECT_ROOT / "tasks" / "A-data-transformation" / "provenance"
TRANSCRIPT_CORRECTIONS_PATH = PROVENANCE_DIRECTORY / "transcript_corrections.json"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smi.data.nxt.parser import has_fatal_errors, parse_corpus  # noqa: E402
from smi.data.nxt.validation import validate_rows, write_validation_report  # noqa: E402


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file without placing environment data in outputs."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write C1 JSONL with deterministic JSON and LF line endings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        for row in rows:
            serialized = json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            output_file.write(serialized)
            output_file.write("\n")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the parser script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="directory containing DialogueActs, Words, and transcripts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination for the generated C1 JSONL",
    )
    return parser


def write_provenance(output_hash: str) -> None:
    """Record the generated artifact path and content hash."""
    PROVENANCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    provenance_path = PROVENANCE_DIRECTORY / "c1.json"
    provenance_path.write_text(
        json.dumps(
            {
                "path": "data/interim/parsed/rows.jsonl",
                "sha256": output_hash,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_transcript_corrections() -> dict[str, list[str]]:
    """Load reviewed transcript-only corrections used for pipe alignment."""
    if not TRANSCRIPT_CORRECTIONS_PATH.exists():
        return {}
    return json.loads(TRANSCRIPT_CORRECTIONS_PATH.read_text(encoding="utf-8"))


def main() -> int:
    arguments = build_argument_parser().parse_args()

    rows, source_counts, transcript_problems = parse_corpus(
        arguments.raw_root,
        load_transcript_corrections(),
    )
    contract_problems = validate_rows(rows)
    write_jsonl(rows, arguments.output)
    output_hash = sha256(arguments.output)

    write_validation_report(
        REPORT_PATH,
        rows,
        source_counts,
        transcript_problems,
        contract_problems,
    )
    write_provenance(output_hash)

    if has_fatal_errors(rows, transcript_problems, contract_problems):
        print(
            "C1 was written for inspection, but validation found fatal "
            "source or contract errors.",
            file=sys.stderr,
        )
        return 1

    print(f"Wrote {len(rows)} C1 rows to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
