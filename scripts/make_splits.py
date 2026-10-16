"""Create a reproducible meeting-level split and fold provenance file."""

from __future__ import annotations

import argparse
from pathlib import Path

from smi.data.splits import (
    DEFAULT_SPLIT_COUNTS,
    DEFAULT_SPLIT_SEED,
    assign_stratified_splits,
    build_split_provenance,
    load_meeting_ids,
    write_split_provenance,
)


def make_splits(
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    seed: str = DEFAULT_SPLIT_SEED,
) -> dict[str, object]:
    """Build split provenance from metadata IDs using the selected split policy."""
    meeting_ids = load_meeting_ids(metadata_path)
    split_meetings = assign_stratified_splits(
        meeting_ids,
        DEFAULT_SPLIT_COUNTS,
        seed=seed,
    )

    reasons = {}
    for split, meetings in split_meetings.items():
        for meeting in meetings:
            series = "".join(character for character in meeting if character.isalpha())
            reasons[meeting] = (
                f"Project-selected 51/12/12 split, stratified by meeting series "
                f"{series}, deterministic seed {seed}"
            )

    provenance = build_split_provenance(
        meeting_ids,
        split_meetings,
        split_source=(
            "Project-selected 51/12/12 meeting-level split, stratified by "
            f"meeting-series prefix using deterministic SHA-256 ordering; seed={seed}"
        ),
        reasons=reasons,
    )
    provenance["split_assignment"] = {
        "method": "stratified by meeting-series prefix; SHA-256 order within each series",
        "seed": seed,
        "counts": dict(DEFAULT_SPLIT_COUNTS),
        "lead_table_9": False,
    }
    write_split_provenance(provenance, output_path)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Write meeting split and fold provenance.")
    parser.add_argument("--metadata", default="data/ICSI-metadata.xml")
    parser.add_argument(
        "--output",
        default="tasks/A-data-transformation/provenance/split_v1.json",
    )
    parser.add_argument("--seed", default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    provenance = make_splits(args.metadata, args.output, seed=args.seed)
    counts = provenance["split_assignment"]["counts"]
    print(f"Wrote {len(provenance['meetings'])} meeting assignments: {counts}")


if __name__ == "__main__":
    main()
