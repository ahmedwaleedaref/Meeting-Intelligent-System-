"""
TEMPORARY trial of the phase-C pipeline (not the PROTOCOL run.py : its numbers are not reportable).

--mode flat         : one 7-class classifier
--mode hierarchical : stage 1 ("other" vs "not_other") + stage 2 (the 6 target classes , trained on non-"other"
                      segments) , combined into one 7-class prediction (smi.models.hierarchical.combine)

--family encoder (default) : full fine-tuning , exactly as without the flag
--family decoder           : DECODER_SETTINGS below : suffix , LoRA , frozen base in fp16 (task C2)
                             --full-finetuning trains every parameter instead (fp32 base)

for each seed : train -> write the best epoch's val predictions -> evaluate (7 classes , the harness) ->
one folder per seed with settings.json , history.jsonl and parameters.json (per stage) , predictions.jsonl ,
metrics.json.

one process , one device (the GPU if there is one). fp16 and fused AdamW are settings below
(measured one at a time on a T4 in concepts/performance-analysis-and-optimization/05).

usage (from the repo root) :
    PYTHONPATH=src python scripts/trial_run.py --mode flat --data-root data/v4_meetings --model microsoft/deberta-v3-base --out-dir runs
    PYTHONPATH=src python scripts/trial_run.py --mode flat --family decoder --lr 2e-4 --epochs 3 --data-root data/v4_meetings --model Qwen/Qwen2.5-0.5B --out-dir runs
"""

import argparse
import importlib.metadata
import json
import math
import statistics
import subprocess
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
from smi.models.model import (
    build_optimizer,
    build_scheduler,
    classification_report,
    count_parameters,
    load_model_and_tokenizer,
    train,
)

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
    "grad_checkpointing": False,  # True = recompute each layer's activations in backward : less memory , ~1/3 more time
    "family": "encoder",      # "encoder" or "decoder" (DECODER_SETTINGS)
    "suffix": "",             # text glued after the last segment ; decoder only
    "base_dtype": "fp32",     # dtype of the loaded weights ; "fp16" only for a frozen base under LoRA
    "load_in_4bit": False,    # frozen base in 4-bit (QLoRA) , when fp16 does not fit
    "lora": None,             # None = full fine-tuning
}

# what --family decoder changes. the suffix wording and the LoRA settings stay fixed across decoder runs and sizes
DECODER_SETTINGS = {
    "family": "decoder",
    "suffix": "\nLabel of [T]:",
    "base_dtype": "fp16",
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
}

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def load_model(model_name: str, labels: tuple[str, ...] = LABELS):
    """load_model_and_tokenizer as SETTINGS say ; the notebooks' memory check uses it too."""
    return load_model_and_tokenizer(
        model_name, labels,
        dtype=DTYPES[SETTINGS["base_dtype"]],
        lora=SETTINGS["lora"],
        load_in_4bit=SETTINGS["load_in_4bit"],
        gradient_checkpointing=SETTINGS["grad_checkpointing"],
    )


def run_environment() -> dict:
    """the commit and library versions of this run (Kaggle keeps its own torch build , so they are not all pinned)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:  # no git on this machine
        commit = ""

    versions = {}
    for package in ("torch", "transformers", "peft"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:  # peft : the encoder runs without it
            versions[package] = None
    return {"commit": commit, "versions": versions}


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
    # one seed for everything random : head initialisation , LoRA's A matrices , dropout and the train batches
    set_seed(seed)
    model, tokenizer = load_model(model_name, labels)
    trainable, total = count_parameters(model)
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
        "suffix": SETTINGS["suffix"],
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
          f"{total_steps} optimizer steps | fp16 {SETTINGS['fp16']} | fused AdamW {SETTINGS['fused_optimizer']} | "
          f"grad checkpointing {SETTINGS['grad_checkpointing']}\n"
          f"  family {SETTINGS['family']} | base {SETTINGS['base_dtype']} | lora {SETTINGS['lora']} | "
          f"trainable parameters {trainable:,} of {total:,} ({100 * trainable / total:.2f} %)",
          flush=True)

    stage_dir.mkdir(parents=True)
    with open(stage_dir / "parameters.json", "x", encoding="utf-8") as f:
        json.dump({"trainable": trainable, "total": total, "fraction": trainable / total}, f, indent=2)

    # every history line is printed and appended to history.jsonl as soon as train creates it
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

    print(f"[{stage_dir.name}] selected epoch {selected_epoch} , truncated train inputs = {collator.n_truncated} , "
          f"truncated val inputs = {val_loader.collate_fn.n_truncated}", flush=True)

    # free the GPU before the next model
    del model, optimizer, scheduler, train_loader
    torch.cuda.empty_cache()
    return selected_epoch, best_records


def run_one_seed(mode: str, seed: int, data_root: Path, model_name: str, run_dir: Path) -> dict:
    """returns the metrics dict (metrics.json)."""
    run_dir.mkdir(parents=True)  # fails if the folder already exists : a run is never overwritten
    with open(run_dir / "settings.json", "x", encoding="utf-8") as f:
        json.dump({"mode": mode, "seed": seed, "model": model_name, "data_root": str(data_root), **SETTINGS,
                   **run_environment()}, f, indent=2)

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


def parse_arguments() -> argparse.Namespace:
    """the command line ; the notebooks' memory check takes the same flags as the run it checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["flat", "hierarchical"], required=True)
    parser.add_argument("--data-root", required=True, help="folder with train/ val/ and source.json")
    parser.add_argument("--model", required=True, help="Hub name or local folder of the checkpoint")
    parser.add_argument("--out-dir", required=True, help="where the per-seed run folders are created")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--name", default=None, help="prefix of the run folders (default: trial_<mode>_k0)")
    parser.add_argument("--epochs", type=int, default=None, help="override SETTINGS['epochs']")
    parser.add_argument("--lr", type=float, default=None, help="override SETTINGS['lr']")
    parser.add_argument("--past-k", type=int, default=None, help="override SETTINGS['past_k'] (context segments before)")
    parser.add_argument("--future-k", type=int, default=None, help="override SETTINGS['future_k'] (context segments after)")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="recompute activations in backward : for inputs too long for the GPU's memory at this batch size")
    parser.add_argument("--fp32", action="store_true",
                        help="fp16 and fused AdamW off : the pre-optimization setup (e.g. to reproduce the first seed-42 run)")
    parser.add_argument("--family", choices=["encoder", "decoder"], default="encoder",
                        help="decoder = DECODER_SETTINGS : suffix , LoRA , frozen base in fp16")
    parser.add_argument("--full-finetuning", action="store_true",
                        help="decoder without LoRA : every parameter trains , base in fp32 (the control of task C2)")
    parser.add_argument("--base-dtype", choices=sorted(DTYPES), default=None, help="override SETTINGS['base_dtype']")
    parser.add_argument("--load-in-4bit", action="store_true", help="frozen base in 4-bit (QLoRA) : needs LoRA and bitsandbytes")
    parser.add_argument("--lora-r", type=int, default=None, help="override the LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="override LoRA's alpha ; report it with r and the LR")
    parser.add_argument("--lora-targets", nargs="+", default=None,
                        help="override the LoRA target modules : module names , or the one word all-linear")
    return parser.parse_args()


def apply_arguments(args: argparse.Namespace) -> None:
    """write the command line into SETTINGS ; invalid combinations stop here , not after a wrong run."""
    if args.epochs is not None:
        SETTINGS["epochs"] = args.epochs
    if args.lr is not None:
        SETTINGS["lr"] = args.lr
    if args.past_k is not None:
        SETTINGS["past_k"] = args.past_k
    if args.future_k is not None:
        SETTINGS["future_k"] = args.future_k
    if args.grad_checkpointing:
        SETTINGS["grad_checkpointing"] = True
    if args.fp32:
        SETTINGS["fp16"] = False
        SETTINGS["fused_optimizer"] = False

    if args.family == "decoder":
        SETTINGS.update(DECODER_SETTINGS)
    if args.full_finetuning:
        if args.family != "decoder":
            raise ValueError("--full-finetuning needs --family decoder : the encoder always trains fully")
        SETTINGS["lora"] = None
        SETTINGS["base_dtype"] = "fp32"
    if args.base_dtype is not None:
        SETTINGS["base_dtype"] = args.base_dtype
    if args.load_in_4bit:
        SETTINGS["load_in_4bit"] = True

    # "all-linear" is a PEFT keyword : a plain string , not a list of module names
    targets = args.lora_targets
    if targets == ["all-linear"]:
        targets = "all-linear"
    lora_overrides = {"r": args.lora_r, "alpha": args.lora_alpha, "target_modules": targets}
    for key in lora_overrides:
        if lora_overrides[key] is not None:
            if SETTINGS["lora"] is None:
                raise ValueError("the --lora-* flags need LoRA : --family decoder , without --full-finetuning")
            SETTINGS["lora"] = {**SETTINGS["lora"], key: lora_overrides[key]}

    if SETTINGS["base_dtype"] == "fp16" and SETTINGS["lora"] is None:
        raise ValueError("an fp16 base cannot train : every trained weight must be fp32 , so it needs LoRA")
    if (SETTINGS["base_dtype"] == "fp16" or SETTINGS["load_in_4bit"]) and not SETTINGS["fp16"]:
        raise ValueError("an fp16 or 4-bit base needs fp16 autocast : the forward fails without it (drop --fp32)")


def main() -> None:
    args = parse_arguments()
    apply_arguments(args)
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
