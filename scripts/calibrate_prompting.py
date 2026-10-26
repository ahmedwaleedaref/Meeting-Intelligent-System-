"""
TEMPORARY : contextual calibration of a finished prompting run (scripts/trial_prompting.py) , without
re-scoring the split. reads the run's scores.jsonl , measures the model's letter bias on content-free
inputs with the run's own prompt settings , divides it out , and grades the result with the harness.

output folder runs/<run>_cc/ : settings.json (source run + measured bias) , predictions.jsonl , metrics.json

usage (from the repo root) :
    PYTHONPATH=src python scripts/calibrate_prompting.py --run runs/c3_exp01_zs_k0_defs --data-root data/v3_meetings
"""

import argparse
import json
from pathlib import Path

from smi.eval.evaluate import evaluate
from smi.eval.prediction_schema import write_predictions
from smi.labels import LABELS
from smi.prompting.backend import load_model
from smi.prompting.calibrate import calibrate, content_free_probs
from smi.prompting.render import load_prompt
from smi.prompting.shots import pick_shots


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a finished trial_prompting.py run folder")
    parser.add_argument("--data-root", required=True, help="the same data the run used , e.g. data/v3_meetings")
    args = parser.parse_args(argv)

    run = Path(args.run)
    data_root = Path(args.data_root)
    settings = json.loads((run / "settings.json").read_text(encoding="utf-8"))
    run_args = settings["args"]  # the source run's own settings : model , prompt files , k , shots

    # the data must be the version the run was scored on
    source = json.loads((data_root / "source.json").read_text(encoding="utf-8"))
    if source["data_version"] != settings["data_version"]:
        raise SystemExit(f"run used data {settings['data_version']}, --data-root is {source['data_version']}")
    if run_args["limit"] is not None:
        raise SystemExit("the run scored only part of the split (--limit) : calibrate a full run")

    # rebuild the run's exact prompt pieces
    template, options = load_prompt(run_args["template"]), load_prompt(run_args["options"])
    shots = []
    if run_args["shots_per_class"] > 0:
        shots = pick_shots(data_root / "train", run_args["past_k"], run_args["future_k"],
                           run_args["shots_per_class"], run_args["shot_seed"])
        if [shot["seg_id"] for shot in shots] != settings["shot_seg_ids"]:
            raise SystemExit("the few-shot examples differ from the run's : different data?")

    # measure the bias : 3 short prompts , quick even on a CPU
    model, tokenizer, letter_ids = load_model(run_args["model"], load_in_4bit=run_args["load_in_4bit"])
    bias = content_free_probs(model, tokenizer, letter_ids, template, options, shots,
                              run_args["past_k"], run_args["future_k"])
    print("bias (content-free probabilities):", {label: round(b, 3) for label, b in zip(LABELS, bias)})

    # calibrate every saved score ; the label is the highest calibrated probability
    records = {}
    with open(run / "scores.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            probs = calibrate(row["probs"], bias)
            records[row["seg_id"]] = {"label": LABELS[probs.index(max(probs))], "scores": dict(zip(LABELS, probs))}

    out = run.parent / (run.name + "_cc")
    out.mkdir()  # fails if it exists : run outputs are never overwritten
    (out / "settings.json").write_text(json.dumps({
        "source_run": str(run), "method": "contextual calibration", "content_free_inputs": ["N/A", "", "[MASK]"],
        "bias": dict(zip(LABELS, bias)), "data_version": settings["data_version"],
    }, indent=2) + "\n", encoding="utf-8")
    write_predictions(records, out / "predictions.jsonl")
    result = evaluate(out / "predictions.jsonl", data_root / run_args["split"], out / "metrics.json",
                      extra={"calibration": {"source_run": str(run), "bias": dict(zip(LABELS, bias))}})

    per_class = {label: round(result["per_class"][label]["f1"], 3) for label in LABELS}
    predicted = {label: sum(1 for r in records.values() if r["label"] == label) for label in LABELS}
    print(f"macro-F1 (6 targets) = {result['primary']['macro_f1_6']:.4f}")
    print(f"per-class F1 = {per_class}")
    print(f"predicted labels = {predicted}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
