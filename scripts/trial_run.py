"""
TEMPORARY trial of the encoder pipeline (not the PROTOCOL run.py : its numbers are not reportable).

--mode flat         : one 7-class classifier
--mode hierarchical : stage 1 ("other" vs "not_other") + stage 2 (the 6 target classes , trained on non-"other"
                      segments) , combined into one 7-class prediction (smi.models.hierarchical.combine)

for each seed : train -> write the best epoch's val predictions -> evaluate (7 classes , the harness) ->
one folder per seed with settings.json , history.jsonl (per stage) , predictions.jsonl , metrics.json.

one process , one device (the GPU if there is one). fp16 and fused AdamW are settings below
(measured one at a time on a T4 in concepts/performance-analysis-and-optimization/05).

usage (from the repo root) :
    PYTHONPATH=src python scripts/trial_run.py --mode flat --data-root data/v4_meetings --model microsoft/deberta-v3-base --out-dir runs
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from transformers import set_seed

from smi.eval.evaluate import evaluate
from smi.eval.gold import load_gold
from smi.eval.prediction_schema import write_predictions
from smi.inputs.loader import build_dataloader
from smi.labels import LABELS, TARGET_LABELS
from smi.models import hierarchical
from smi.models.model import build_optimizer, build_scheduler, classification_report, load_model_and_tokenizer, train

SETTINGS = {
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
    "fp16": True,             # mixed precision (GPU only) ; False = fp32 , the pre-optimization setup
    "fused_optimizer": True,  # fused AdamW ; False = PyTorch's default implementation
}


def run_stage(
    seed: int,
    stage_dir: Path,
    model_name: str,
    data_root: Path,
    labels: tuple[str, ...],
    select_labels: tuple[str, ...],
    label_map: list[int] | None,
    train_keep_labels: tuple[str, ...] | None,
    val_gold: dict[str, str],
) -> tuple[int, dict[str, dict]]:
    """train one classifier ; returns (selected epoch , its val records)."""
    # one seed for everything random : head initialisation , dropout and the train batches
    set_seed(seed)
    model, tokenizer = load_model_and_tokenizer(model_name, labels)
    # on its device before the optimizer is built : fused AdamW needs the parameters on the GPU
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    loader_args = {
        "tokenizer": tokenizer,
        "past_k": SETTINGS["past_k"],
        "future_k": SETTINGS["future_k"],
        "chunk_batches": SETTINGS["chunk_batches"],
        "max_length": SETTINGS["max_length"],
        "separator": SETTINGS["separator"],
        "markers": tuple(SETTINGS["markers"]),
    }
    train_loader = build_dataloader(
        data_root / "train", batch_size=SETTINGS["batch_size"], shuffle=True, seed=seed,
        keep_labels=train_keep_labels, **loader_args,
    )
    val_loader = build_dataloader(data_root / "val", batch_size=SETTINGS["batch_size"], shuffle=False, **loader_args)
    collator = train_loader.collate_fn

    optimizer = build_optimizer(model, SETTINGS["lr"], SETTINGS["weight_decay"], fused=SETTINGS["fused_optimizer"])
    scheduler = build_scheduler(
        optimizer, train_loader, SETTINGS["epochs"], SETTINGS["warmup_ratio"], SETTINGS["grad_accum_steps"]
    )

    total_steps = SETTINGS["epochs"] * math.ceil(len(train_loader) / SETTINGS["grad_accum_steps"])
    print(f"[{stage_dir.name}] seed {seed}: {len(train_loader)} train batches per epoch , "
          f"{total_steps} optimizer steps | fp16 {SETTINGS['fp16']} | fused AdamW {SETTINGS['fused_optimizer']}",
          flush=True)

    # every history line is printed and appended to history.jsonl as soon as train creates it
    stage_dir.mkdir(parents=True)
    history_file = open(stage_dir / "history.jsonl", "x", encoding="utf-8")

    def on_log(line: dict) -> None:
        history_file.write(json.dumps(line) + "\n")
        history_file.flush()
        if line["type"] == "step":
            print(f"  step {line['step']}/{total_steps} (epoch {line['epoch']}) | loss {line['train_loss']:.4f} "
                  f"| lr {line['lr']:.2e} | grad_norm {line['grad_norm']:.2f} | "
                  + (f"loss scale {line['loss_scale']:.0f} | " if line["loss_scale"] is not None else "")
                  + f"{line['seconds'] / 60:.1f} min", flush=True)
        else:
            per_class = " ".join(f"{label} {f1:.3f}" for label, f1 in line["per_class_f1"].items())
            print(f"EPOCH {line['epoch']} | train_loss {line['train_loss']:.4f} | val macro F1 {line['val_macro_f1']:.4f} "
                  f"| best {line['is_best']} | {line['seconds'] / 60:.1f} min\n"
                  f"  per-class F1: {per_class}", flush=True)

    try:
        selected_epoch, best_records, history = train(
            model, train_loader, val_loader, val_gold, optimizer, scheduler,
            SETTINGS["epochs"], SETTINGS["grad_accum_steps"], SETTINGS["max_grad_norm"], SETTINGS["log_every"],
            labels=labels, select_labels=select_labels, label_map=label_map, on_log=on_log, fp16=SETTINGS["fp16"],
        )
    finally:
        history_file.close()

    print(f"[{stage_dir.name}] selected epoch {selected_epoch} , truncated train inputs = {collator.n_truncated}",
          flush=True)

    # free the GPU before the next model
    del model, optimizer, scheduler, train_loader
    torch.cuda.empty_cache()
    return selected_epoch, best_records


def run_one_seed(mode: str, seed: int, data_root: Path, model_name: str, run_dir: Path) -> dict:
    """returns the metrics dict (metrics.json)."""
    run_dir.mkdir(parents=True)  # fails if the folder already exists : a run is never overwritten
    with open(run_dir / "settings.json", "x", encoding="utf-8") as f:
        json.dump({"mode": mode, "seed": seed, "model": model_name, "data_root": str(data_root), **SETTINGS}, f, indent=2)

    gold = load_gold(data_root / "val")
    if mode == "flat":
        selected_epoch, records = run_stage(
            seed, run_dir / "flat", model_name, data_root,
            labels=LABELS, select_labels=TARGET_LABELS, label_map=None, train_keep_labels=None, val_gold=gold,
        )
        extra = {"selected_epoch": selected_epoch, "seed": seed, "mode": mode}
    else:
        stage1_epoch, stage1_records = run_stage(
            seed, run_dir / "stage1", model_name, data_root,
            labels=hierarchical.STAGE1_LABELS, select_labels=(hierarchical.NOT_OTHER,),
            label_map=hierarchical.stage1_label_map(), train_keep_labels=None,
            val_gold=hierarchical.stage1_gold(gold),
        )
        stage2_epoch, stage2_records = run_stage(
            seed, run_dir / "stage2", model_name, data_root,
            labels=hierarchical.STAGE2_LABELS, select_labels=hierarchical.STAGE2_LABELS,
            label_map=hierarchical.stage2_label_map(), train_keep_labels=hierarchical.STAGE2_LABELS,
            val_gold=hierarchical.stage2_gold(gold),
        )
        # each stage alone : stage 1 on every val segment , stage 2 on the val segments whose gold is a target class
        stage_reports = [
            ("metrics_stage1.json", stage1_records, hierarchical.stage1_gold(gold),
             hierarchical.STAGE1_LABELS, (hierarchical.NOT_OTHER,), stage1_epoch),
            ("metrics_stage2.json", stage2_records, hierarchical.stage2_gold(gold),
             hierarchical.STAGE2_LABELS, hierarchical.STAGE2_LABELS, stage2_epoch),
        ]
        for file_name, stage_records, stage_gold, stage_labels, stage_select, stage_epoch in stage_reports:
            report = classification_report(stage_records, stage_gold, stage_labels, stage_select)
            report["selected_epoch"] = stage_epoch
            with open(run_dir / file_name, "x", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            per_class = " ".join(f"{label} P {m['p']:.3f} R {m['r']:.3f} F1 {m['f1']:.3f} |"
                                 for label, m in report["per_class"].items())
            print(f"{file_name}: macro F1 {report['macro_f1']:.4f}\n  {per_class}", flush=True)

        # both combined : one 7-class prediction per segment , evaluated by the harness below (metrics.json)
        records = hierarchical.combine(stage1_records, stage2_records)
        extra = {"selected_epoch_stage1": stage1_epoch, "selected_epoch_stage2": stage2_epoch,
                 "seed": seed, "mode": mode}

    write_predictions(records, run_dir / "predictions.jsonl")
    metrics = evaluate(run_dir / "predictions.jsonl", data_root / "val", run_dir / "metrics.json", extra=extra)
    per_class = " ".join(f"{label} {m['f1']:.3f}" for label, m in metrics["per_class"].items())
    print(f"seed {seed} ({mode}): val macro_f1_6 = {metrics['primary']['macro_f1_6']:.4f}\n  per-class F1: {per_class}",
          flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["flat", "hierarchical"], required=True)
    parser.add_argument("--data-root", required=True, help="folder with train/ val/ and source.json")
    parser.add_argument("--model", required=True, help="Hub name or local folder of the checkpoint")
    parser.add_argument("--out-dir", required=True, help="where the per-seed run folders are created")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--name", default=None, help="prefix of the run folders (default: trial_<mode>_k0)")
    parser.add_argument("--epochs", type=int, default=None, help="override SETTINGS['epochs']")
    parser.add_argument("--lr", type=float, default=None, help="override SETTINGS['lr']")
    parser.add_argument("--fp32", action="store_true",
                        help="fp16 and fused AdamW off : the pre-optimization setup (e.g. to reproduce the first seed-42 run)")
    args = parser.parse_args()
    if args.epochs is not None:
        SETTINGS["epochs"] = args.epochs
    if args.lr is not None:
        SETTINGS["lr"] = args.lr
    if args.fp32:
        SETTINGS["fp16"] = False
        SETTINGS["fused_optimizer"] = False
    name = args.name if args.name is not None else f"trial_{args.mode}_k0"

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    scores = []
    for seed in args.seeds:
        start = time.perf_counter()
        metrics = run_one_seed(args.mode, seed, data_root, args.model, out_dir / f"{name}_seed{seed}")
        scores.append(metrics["primary"]["macro_f1_6"])
        print(f"seed {seed} took {(time.perf_counter() - start) / 60:.1f} min", flush=True)

    print("val macro_f1_6 per seed:", [round(s, 4) for s in scores])
    if len(scores) > 1:
        print(f"mean {statistics.mean(scores):.4f} , std {statistics.stdev(scores):.4f}")


if __name__ == "__main__":
    main()
