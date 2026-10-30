"""
TEMPORARY : PriDe (Zheng et al. 2024 , src/smi/prompting/pride.py) for finished prompting runs , without re-scoring.
takes the 7 runs of one setting scored with --option-shift 0 , 1 , ... , 6 (scripts/trial_prompting.py) and writes :
    <name>_cyclic/      every segment's scores averaged over the 7 rotations (the paper's equation 1 , 7 × the cost)
    <name>_pride<d>/    PriDe : the letter prior estimated on a random --share of the segments (their rotations
                        averaged) , divided out of every other segment's unrotated scores ; one folder per draw d
each folder : settings.json (source runs , prior , draw) , predictions.jsonl , metrics.json (the harness).
no labels are used before grading : the prior comes from the segments' scores only.

usage (from the repo root) :
    PYTHONPATH=src python scripts/pride_prompting.py --data-root data/v3_meetings --name runs/c3_exp10_7b_zs \
        --run runs/c3_exp10_7b_zs_r0 runs/c3_exp10_7b_zs_r1 ... runs/c3_exp10_7b_zs_r6
"""

import argparse
import json
import random
from pathlib import Path

from smi.eval.evaluate import evaluate
from smi.eval.prediction_schema import write_predictions
from smi.labels import LABELS
from smi.prompting.pride import average, letter_prior, pride
from smi.prompting.render import LETTERS

# settings that may differ between the 7 runs of one setting ; everything else must be identical
PER_RUN_ARGS = ("run_name", "option_shift")


def shared_settings(settings: dict) -> dict:
    """a run's settings.json without the settings that differ between the 7 runs of one setting."""
    return {**settings, "args": {k: v for k, v in settings["args"].items() if k not in PER_RUN_ARGS}}


def load_rotations(runs: list[Path], data_root: Path) -> tuple[dict[str, list[list[float]]], dict]:
    """
    every segment's probabilities under each rotation , checked to come from one setting.
    returns (seg_id -> probabilities (LABELS order) for shift 0 .. 6 , the shift-0 run's settings)
    """
    source = json.loads((data_root / "source.json").read_text(encoding="utf-8"))
    by_shift: dict[int, tuple[Path, dict]] = {}
    for run in runs:
        settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
        args = settings["args"]
        if settings["data_version"] != source["data_version"]:
            raise SystemExit(f"{run}: run used data {settings['data_version']}, --data-root is {source['data_version']}")
        if args["limit"] is not None:
            raise SystemExit(f"{run}: the run scored only part of the split (--limit)")
        by_shift[args.get("option_shift", 0)] = (run, settings)
    if len(runs) != len(LETTERS) or sorted(by_shift) != list(range(len(LETTERS))):
        raise SystemExit(f"expected one run per --option-shift 0..{len(LETTERS) - 1}, got {len(runs)} runs")
    reference = shared_settings(by_shift[0][1])
    for run, settings in by_shift.values():
        if shared_settings(settings) != reference:
            raise SystemExit(f"{run}: different settings from the shift-0 run (only {PER_RUN_ARGS} may differ)")

    rotated: dict[str, list[list[float]]] = {}
    for shift in range(len(LETTERS)):
        run = by_shift[shift][0]
        with open(run / "scores.jsonl", encoding="utf-8") as f:
            probs = {row["seg_id"]: row["probs"] for row in map(json.loads, f)}
        if shift > 0 and probs.keys() != rotated.keys():
            raise SystemExit(f"{run}: scored other segments than the shift-0 run")
        for seg_id, p in probs.items():
            rotated.setdefault(seg_id, []).append(p)
    return rotated, by_shift[0][1]


def write_debiased(out: Path, probs: dict[str, list[float]], settings: dict, split_dir: Path) -> float:
    """predictions (label = highest debiased probability) and the harness's metrics ; returns the macro-F1."""
    records = {seg_id: {"label": LABELS[p.index(max(p))], "scores": dict(zip(LABELS, p))} for seg_id, p in probs.items()}
    out.mkdir()  # fails if it exists : run outputs are never overwritten
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    write_predictions(records, out / "predictions.jsonl")
    result = evaluate(out / "predictions.jsonl", split_dir, out / "metrics.json", extra={"debiasing": settings})
    per_class = {label: round(result["per_class"][label]["f1"], 3) for label in LABELS}
    predicted = {label: sum(1 for r in records.values() if r["label"] == label) for label in LABELS}
    print(f"{out.name}: macro-F1 (6 targets) = {result['primary']['macro_f1_6']:.4f}")
    print(f"  per-class F1 = {per_class}")
    print(f"  predicted labels = {predicted}")
    return result["primary"]["macro_f1_6"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, nargs="+", help="the 7 run folders , --option-shift 0 .. 6 , any order")
    parser.add_argument("--data-root", required=True, help="the same data the runs used , e.g. data/v3_meetings")
    parser.add_argument("--name", required=True, help="output folders <name>_cyclic , <name>_pride<d> , e.g. runs/c3_exp10_7b_zs")
    parser.add_argument("--share", type=float, default=0.05, help="share of segments the prior is estimated on (paper: 5 %%)")
    parser.add_argument("--draws", type=int, default=5, help="random draws of that share (paper: 5)")
    parser.add_argument("--seed", type=int, default=0, help="draw d uses seed + d")
    args = parser.parse_args(argv)

    data_root, name = Path(args.data_root), Path(args.name)
    outputs = [name.parent / f"{name.name}_cyclic"] + [name.parent / f"{name.name}_pride{d}" for d in range(args.draws)]
    for out in outputs:
        if out.exists():
            raise SystemExit(f"{out} exists : run outputs are never overwritten")
    rotated, settings = load_rotations([Path(run) for run in args.run], data_root)
    split_dir = data_root / settings["args"]["split"]
    source = {"source_runs": [str(run) for run in args.run], "data_version": settings["data_version"]}

    # the letter prior over every segment : how the model's letter preference looks , for the record only
    overall = average([letter_prior(scores) for scores in rotated.values()])
    print("letter prior over all segments:", {letter: round(p, 4) for letter, p in zip(LETTERS, overall)})

    write_debiased(outputs[0], {seg_id: average(scores) for seg_id, scores in rotated.items()},
                   {**source, "method": "cyclic permutation (equation 1)", "letter_prior_all_segments": dict(zip(LETTERS, overall))},
                   split_dir)

    seg_ids = sorted(rotated)
    n_estimation = max(1, round(args.share * len(seg_ids)))
    scores = []
    for draw, out in enumerate(outputs[1:]):
        estimation = set(random.Random(args.seed + draw).sample(seg_ids, n_estimation))
        debiased, prior = pride(rotated, estimation)
        print(f"draw {draw} prior ({n_estimation} segments):", {letter: round(p, 4) for letter, p in zip(LETTERS, prior)})
        scores.append(write_debiased(out, debiased, {
            **source, "method": "PriDe (algorithm 1)", "share": args.share, "n_estimation": n_estimation,
            "seed": args.seed + draw, "prior": dict(zip(LETTERS, prior)),
        }, split_dir))
    print(f"PriDe over {args.draws} draws: macro-F1 mean {sum(scores) / len(scores):.4f} , "
          f"min {min(scores):.4f} , max {max(scores):.4f}")


if __name__ == "__main__":
    main()
