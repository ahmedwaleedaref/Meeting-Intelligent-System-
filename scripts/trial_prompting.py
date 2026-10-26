"""
TEMPORARY driver for the prompting baseline (C3) , until the PROTOCOL run.py exists : its numbers are trial runs.

one run = one prompt setting on one split :
    every segment -> prompt -> the model's 7 letter probabilities -> predictions.jsonl -> evaluate() -> metrics.json

output folder runs/<run-name>/ :
    settings.json      every setting of the run , the data version , the prompt files' hashes , the few-shot seg_ids
    scores.jsonl       one line per scored segment , written as it goes : if the run stops (e.g. Kaggle's 12 h cap) ,
                       starting the same command again skips the segments already scored
    predictions.jsonl  the harness's prediction file
    metrics.json       the harness's metrics , plus a "prompting" block (letter mass , time)

usage (from the repo root) :
    PYTHONPATH=src python scripts/trial_prompting.py --data-root data/v3_meetings --run-name c3_exp01_k0_defs
    add --limit 50 for a quick check on the first 50 segments (metrics then cover those segments only)
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from smi.eval.evaluate import compute_metrics, evaluate
from smi.eval.gold import load_gold
from smi.eval.prediction_schema import write_predictions
from smi.inputs.dataset import SegmentDataset
from smi.labels import LABELS
from smi.prompting.backend import load_model, score_batch
from smi.prompting.render import PROMPTS_DIR, load_prompt, render_prompt
from smi.prompting.shots import pick_shots


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="folder with train/ val/ source.json , e.g. data/v3_meetings")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="test is never used here (PROTOCOL §7)")
    parser.add_argument("--run-name", required=True, help="output folder name under --out-dir")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true", help="4-bit weights (GPU + bitsandbytes) for large models")
    parser.add_argument("--template", default="template_v1.txt", help="file in configs/prompts/")
    parser.add_argument("--options", default="definitions_v1.txt", help="definitions_v1.txt (on) or names_v1.txt (off)")
    parser.add_argument("--past-k", type=int, default=0)
    parser.add_argument("--future-k", type=int, default=0)
    parser.add_argument("--shots-per-class", type=int, default=0, help="0 = zero-shot")
    parser.add_argument("--shot-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N segments (quick checks)")
    return parser.parse_args(argv)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    """the commit the code runs from , with "-dirty" if there are uncommitted changes ; None outside git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> None:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    out = Path(args.out_dir) / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. inputs : segments , prompt files , few-shot examples
    dataset = SegmentDataset(split_dir, past_k=args.past_k, future_k=args.future_k)
    items = [dataset[i] for i in range(len(dataset))]
    if args.limit is not None:
        items = items[: args.limit]
    template, options = load_prompt(args.template), load_prompt(args.options)
    shots = []
    if args.shots_per_class > 0:
        shots = pick_shots(data_root / "train", args.past_k, args.future_k, args.shots_per_class, args.shot_seed)

    # ---- 2. settings.json : what this run is. a restarted run must have exactly the same settings
    source = json.loads((data_root / "source.json").read_text(encoding="utf-8"))
    settings = {
        "args": vars(args),
        "data_version": source["data_version"],
        "template_sha256": sha256_text(template),
        "options_sha256": sha256_text(options),
        "shot_seg_ids": [shot["seg_id"] for shot in shots],
    }
    settings_path = out / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != settings:
            raise SystemExit(f"{settings_path} exists with different settings : use another --run-name")
    else:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    # ---- 3. resume : segments already in scores.jsonl are not scored again
    scores_path = out / "scores.jsonl"
    done: dict[str, dict] = {}
    if scores_path.exists():
        with open(scores_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done[row["seg_id"]] = row
    todo = [item for item in items if item["seg_id"] not in done]
    print(f"{len(items)} segments , {len(done)} already scored , {len(todo)} to score", flush=True)

    # ---- 4. score : prompts sorted longest first (similar lengths share a batch = little padding ;
    #         an out-of-memory error shows up in the first batch , not after an hour)
    if todo:
        model, tokenizer, letter_ids = load_model(args.model, load_in_4bit=args.load_in_4bit)
        prompts = {item["seg_id"]: render_prompt(item, template, options, shots) for item in todo}
        todo.sort(key=lambda item: len(prompts[item["seg_id"]]), reverse=True)
        start = time.time()
        with open(scores_path, "a", encoding="utf-8") as f:
            for b in range(0, len(todo), args.batch_size):
                batch = todo[b : b + args.batch_size]
                results = score_batch(model, tokenizer, letter_ids, [prompts[item["seg_id"]] for item in batch])
                for item, result in zip(batch, results):
                    row = {"seg_id": item["seg_id"], "probs": result["probs"], "letter_mass": result["letter_mass"]}
                    f.write(json.dumps(row) + "\n")
                    done[item["seg_id"]] = row
                f.flush()  # on disk now : survives a crash
                n = b + len(batch)
                if (b // args.batch_size) % 50 == 0 or n == len(todo):
                    rate = n / (time.time() - start)
                    print(f"  {n}/{len(todo)}  {rate:.1f} segments/s  ~{(len(todo) - n) / rate / 60:.0f} min left", flush=True)

    # ---- 5. predictions in the harness format : label = the letter with the highest probability
    records = {}
    for item in items:
        probs = done[item["seg_id"]]["probs"]
        best = probs.index(max(probs))
        records[item["seg_id"]] = {"label": LABELS[best], "scores": dict(zip(LABELS, probs))}
    write_predictions(records, out / "predictions.jsonl")

    # ---- 6. metrics through the harness
    masses = [done[item["seg_id"]]["letter_mass"] for item in items]
    extra = {"prompting": {
        "mean_letter_mass": sum(masses) / len(masses),
        "min_letter_mass": min(masses),
        "n_scored": len(items),
        "git_commit": git_commit(),
    }}
    if args.limit is None:
        result = evaluate(out / "predictions.jsonl", split_dir, out / "metrics.json", extra=extra)
    else:
        # a partial split : evaluate() needs every segment of the split , so score the subset in memory
        gold = load_gold(split_dir)
        result = compute_metrics(records, {seg_id: gold[seg_id] for seg_id in records})
        result.update(extra, limit=args.limit, data_version=source["data_version"])
        with open(out / "metrics.json", "x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    per_class = {label: round(result["per_class"][label]["f1"], 3) for label in LABELS}
    print(f"macro-F1 (6 targets) = {result['primary']['macro_f1_6']:.4f}")
    print(f"per-class F1 = {per_class}")
    print(f"mean letter mass = {extra['prompting']['mean_letter_mass']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
