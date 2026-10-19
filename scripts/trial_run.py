"""
TEMPORARY trial of the encoder pipeline (not the PROTOCOL run.py : its numbers are not reportable).
for each seed : load model -> build loaders , optimizer , scheduler -> train -> write the best epoch's val
predictions -> evaluate -> one folder per seed with settings.json , history.jsonl , predictions.jsonl , metrics.json.

usage (from the repo root) :
    PYTHONPATH=src python scripts/trial_run.py --data-root data/v4_meetings --model microsoft/deberta-v3-base --out-dir runs
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from transformers import set_seed

from smi.eval.evaluate import evaluate
from smi.eval.gold import load_gold
from smi.eval.prediction_schema import write_predictions
from smi.inputs.loader import build_dataloader
from smi.models.model import build_optimizer, build_scheduler, load_model_and_tokenizer, train

SETTINGS = {
    "seeds": [42, 43, 44],
    "past_k": 0,
    "future_k": 0,
    "max_length": 512,
    "separator": "\n",
    "markers": ["[T] ", " [/T]"],
    "batch_size": 32,
    "chunk_batches": 50,
    "lr": 2.0e-5,
    "weight_decay": 0.01,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "grad_accum_steps": 1,
    "max_grad_norm": 1.0,
    "log_every": 50,
}


def run_one_seed(seed: int, data_root: Path, model_name: str, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True)  # fails if the folder already exists : a run is never overwritten

    # one seed for everything random : head initialisation , dropout and the train batches
    set_seed(seed)
    model, tokenizer = load_model_and_tokenizer(model_name)

    loader_args = {
        "tokenizer": tokenizer,
        "past_k": SETTINGS["past_k"],
        "future_k": SETTINGS["future_k"],
        "batch_size": SETTINGS["batch_size"],
        "chunk_batches": SETTINGS["chunk_batches"],
        "max_length": SETTINGS["max_length"],
        "separator": SETTINGS["separator"],
        "markers": tuple(SETTINGS["markers"]),
    }
    train_loader = build_dataloader(data_root / "train", shuffle=True, seed=seed, **loader_args)
    val_loader = build_dataloader(data_root / "val", shuffle=False, **loader_args)
    val_gold = load_gold(data_root / "val")

    optimizer = build_optimizer(model, SETTINGS["lr"], SETTINGS["weight_decay"])
    scheduler = build_scheduler(
        optimizer, train_loader, SETTINGS["epochs"], SETTINGS["warmup_ratio"], SETTINGS["grad_accum_steps"]
    )

    with open(run_dir / "settings.json", "x", encoding="utf-8") as f:
        json.dump({"seed": seed, "model": model_name, "data_root": str(data_root), **SETTINGS}, f, indent=2)

    total_steps = SETTINGS["epochs"] * math.ceil(len(train_loader) / SETTINGS["grad_accum_steps"])
    print(f"seed {seed}: {len(train_loader)} train batches per epoch , {total_steps} optimizer steps in total", flush=True)

    # every history line is printed and appended to history.jsonl as soon as train creates it
    history_file = open(run_dir / "history.jsonl", "x", encoding="utf-8")

    def on_log(line: dict) -> None:
        history_file.write(json.dumps(line) + "\n")
        history_file.flush()
        if line["type"] == "step":
            print(f"  step {line['step']}/{total_steps} (epoch {line['epoch']}) | loss {line['train_loss']:.4f} "
                  f"| lr {line['lr']:.2e} | grad_norm {line['grad_norm']:.2f} | {line['seconds'] / 60:.1f} min", flush=True)
        else:
            per_class = " ".join(f"{label} {f1:.3f}" for label, f1 in line["per_class_f1"].items())
            print(f"EPOCH {line['epoch']} | train_loss {line['train_loss']:.4f} | val macro_f1_6 {line['macro_f1_6']:.4f} "
                  f"| best {line['is_best']} | {line['seconds'] / 60:.1f} min\n  per-class F1: {per_class}", flush=True)

    try:
        selected_epoch, best_records, history = train(
            model, train_loader, val_loader, val_gold, optimizer, scheduler,
            SETTINGS["epochs"], SETTINGS["grad_accum_steps"], SETTINGS["max_grad_norm"], SETTINGS["log_every"],
            on_log=on_log,
        )
    finally:
        history_file.close()

    write_predictions(best_records, run_dir / "predictions.jsonl")
    metrics = evaluate(
        run_dir / "predictions.jsonl", data_root / "val", run_dir / "metrics.json",
        extra={"selected_epoch": selected_epoch, "seed": seed},
    )
    print(f"seed {seed}: selected epoch {selected_epoch} , val macro_f1_6 = {metrics['primary']['macro_f1_6']:.4f} , "
          f"truncated inputs (train) = {train_loader.collate_fn.n_truncated}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="folder with train/ val/ and source.json")
    parser.add_argument("--model", required=True, help="Hub name or local folder of the checkpoint")
    parser.add_argument("--out-dir", required=True, help="where the per-seed run folders are created")
    parser.add_argument("--name", default="trial_deberta_k0", help="prefix of the run folders")
    parser.add_argument("--epochs", type=int, default=None, help="override SETTINGS['epochs'] (smoke tests)")
    args = parser.parse_args()
    if args.epochs is not None:
        SETTINGS["epochs"] = args.epochs

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    scores = []
    for seed in SETTINGS["seeds"]:
        start = time.perf_counter()
        metrics = run_one_seed(seed, data_root, args.model, out_dir / f"{args.name}_seed{seed}")
        scores.append(metrics["primary"]["macro_f1_6"])
        print(f"seed {seed} took {(time.perf_counter() - start) / 60:.1f} min", flush=True)

    print("val macro_f1_6 per seed:", [round(s, 4) for s in scores])
    print(f"mean {statistics.mean(scores):.4f} , std {statistics.stdev(scores):.4f}")


if __name__ == "__main__":
    main()
