"""
TEMPORARY : contextual calibration of finished prompting runs (scripts/trial_prompting.py) , without
re-scoring the split. reads each run's scores.jsonl , measures the model's letter bias on content-free
inputs with the run's own prompt settings , divides it out , and grades the result with the harness.
several runs of the same model are calibrated with one model load.

--target : what a content-free input is calibrated to , and the output folder
    uniform      every letter equal (Zhao et al. 2021)       runs/<run>_cc/
    train-prior  the train label frequencies (84 % other)    runs/<run>_ccp/
each output folder : settings.json (source run , measured bias , target) , predictions.jsonl , metrics.json

usage (from the repo root) :
    PYTHONPATH=src python scripts/calibrate_prompting.py --run runs/c3_exp01_zs_k0_defs --data-root data/v3_meetings
    PYTHONPATH=src python scripts/calibrate_prompting.py --run runs/a runs/b --target uniform train-prior --data-root data/v4_meetings
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from smi.eval.evaluate import evaluate
from smi.eval.prediction_schema import write_predictions
from smi.inputs.dataset import SegmentItem
from smi.labels import LABELS
from smi.prompting.backend import load_model
from smi.prompting.calibrate import CONTENT_FREE, calibrate, content_free_probs, label_frequencies
from smi.prompting.render import load_prompt
from smi.prompting.shots import order_shots, pick_shots

# --target -> the suffix of its output folder
TARGETS = {"uniform": "_cc", "train-prior": "_ccp"}


@dataclass(frozen=True)
class SourceRun:
    """a finished run , with the exact prompt pieces it was scored with."""
    path: Path
    settings: dict
    args: dict
    template: str
    options: str
    shots: list[SegmentItem]


def load_run(run: Path, data_root: Path, targets: list[str]) -> SourceRun:
    """read a run's settings and rebuild its prompt pieces ; every check runs here , before the model loads."""
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    run_args = settings["args"]  # the source run's own settings : model , prompt files , k , shots

    # the data must be the version the run was scored on
    source = json.loads((data_root / "source.json").read_text(encoding="utf-8"))
    if source["data_version"] != settings["data_version"]:
        raise SystemExit(f"{run}: run used data {settings['data_version']}, --data-root is {source['data_version']}")
    if run_args["limit"] is not None:
        raise SystemExit(f"{run}: the run scored only part of the split (--limit) : calibrate a full run")
    for target in targets:
        if (run.parent / (run.name + TARGETS[target])).exists():
            raise SystemExit(f"{run.name}{TARGETS[target]} exists : run outputs are never overwritten")

    # rebuild the run's exact prompt pieces ; runs from before --shot-order used the "labels" order
    shots = []
    if run_args["shots_per_class"] > 0:
        shots = pick_shots(data_root / "train", run_args["past_k"], run_args["future_k"],
                           run_args["shots_per_class"], run_args["shot_seed"])
        shots = order_shots(shots, run_args.get("shot_order", "labels"), run_args.get("shot_order_seed", 0))
        if [shot["seg_id"] for shot in shots] != settings["shot_seg_ids"]:
            raise SystemExit(f"{run}: the few-shot examples differ from the run's : different data?")
    return SourceRun(run, settings, run_args, load_prompt(run_args["template"]), load_prompt(run_args["options"]), shots)


def write_calibrated(run: SourceRun, bias: list[float], target: str, target_probs: list[float] | None,
                     data_root: Path) -> None:
    """calibrate every saved score of the run , then write and grade its output folder."""
    # the label is the highest calibrated probability
    records = {}
    with open(run.path / "scores.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            probs = calibrate(row["probs"], bias, target_probs)
            records[row["seg_id"]] = {"label": LABELS[probs.index(max(probs))], "scores": dict(zip(LABELS, probs))}

    out = run.path.parent / (run.path.name + TARGETS[target])
    out.mkdir()  # fails if it exists : run outputs are never overwritten
    calibration = {
        "source_run": str(run.path), "method": "contextual calibration", "content_free_inputs": list(CONTENT_FREE),
        "bias": dict(zip(LABELS, bias)), "target": target,
        "target_probs": None if target_probs is None else dict(zip(LABELS, target_probs)),
    }
    settings = {**calibration, "data_version": run.settings["data_version"]}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    write_predictions(records, out / "predictions.jsonl")
    result = evaluate(out / "predictions.jsonl", data_root / run.args["split"], out / "metrics.json",
                      extra={"calibration": calibration})

    per_class = {label: round(result["per_class"][label]["f1"], 3) for label in LABELS}
    predicted = {label: sum(1 for r in records.values() if r["label"] == label) for label in LABELS}
    print(f"{out.name}: macro-F1 (6 targets) = {result['primary']['macro_f1_6']:.4f}")
    print(f"  per-class F1 = {per_class}")
    print(f"  predicted labels = {predicted}")
    print(f"  wrote {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, nargs="+", help="finished trial_prompting.py run folders , one model")
    parser.add_argument("--data-root", required=True, help="the same data the runs used , e.g. data/v4_meetings")
    parser.add_argument("--target", nargs="+", default=["uniform"], choices=list(TARGETS), help="default: uniform")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    runs = [load_run(Path(run), data_root, args.target) for run in args.run]
    models = {(run.args["model"], run.args["load_in_4bit"]) for run in runs}
    if len(models) != 1:
        raise SystemExit(f"the runs use different models {sorted(models)} : calibrate them in separate calls")
    target_probs = {"uniform": None}
    if "train-prior" in args.target:
        target_probs["train-prior"] = label_frequencies(data_root / "train")
        print("train label frequencies:", {label: round(p, 3) for label, p in zip(LABELS, target_probs["train-prior"])})

    # measure each run's bias : 3 short prompts per run , quick even on a CPU ; the model loads once
    [(name, load_in_4bit)] = models
    model, tokenizer, letter_ids = load_model(name, load_in_4bit=load_in_4bit)
    for run in runs:
        bias = content_free_probs(model, tokenizer, letter_ids, run.template, run.options, run.shots,
                                  run.args["past_k"], run.args["future_k"])
        print(f"{run.path.name} bias (content-free probabilities):", {label: round(b, 3) for label, b in zip(LABELS, bias)})
        for target in args.target:
            write_calibrated(run, bias, target, target_probs[target], data_root)


if __name__ == "__main__":
    main()
