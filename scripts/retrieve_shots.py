"""
TEMPORARY : every val segment's own few-shot examples (KATE , Liu et al. 2021 ; src/smi/prompting/retrieve.py) ,
for scripts/trial_prompting.py --shots-file.

--select similar : per class , the --per-class train examples closest to the segment : cosine between the
                   embeddings of --encoder (a Hub name or a local folder) , pooled with --pooling
    --fine-tune  : first fine-tune the encoder on train as C1 does (scripts/trial_run.py's settings with C1's
                   adopted lr 1e-5 and 3 epochs , C1 exp02 / exp03) , so closeness follows the dialogue acts and
                   not only the words. the LAST epoch is kept : the val labels choose nothing , they are only scored
--select random  : per class , --per-class random train examples for each segment (the control ; no encoder)

examples come from train , one per (class , transcript) (retrieve.example_pool).
output folder --out (never overwritten) : the shots file (settings.json , shots.jsonl) , and
    nearest/        --select similar : each val segment labelled with the class of its most similar example ,
                    the retriever alone without the LLM (KATE's kNN baseline) , graded by the harness
    encoder/        --fine-tune : the fine-tuned encoder's own val predictions , graded by the harness
    history.jsonl   --fine-tune : the encoder's training (C1's train() , val macro-F1 after every epoch)

usage (from the repo root) :
    PYTHONPATH=src python scripts/retrieve_shots.py --data-root data/v3_meetings --out shots/tuned \
        --select similar --encoder microsoft/deberta-v3-base --fine-tune --seed 42 --pooling cls
    PYTHONPATH=src python scripts/retrieve_shots.py --data-root data/v3_meetings --out shots/random --select random
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import set_seed

from smi.eval.evaluate import evaluate
from smi.eval.gold import load_gold
from smi.eval.prediction_schema import write_predictions
from smi.inputs.dataset import SegmentDataset
from smi.inputs.loader import build_dataloader
from smi.labels import LABELS
from smi.models.model import build_optimizer, build_scheduler, load_model_and_tokenizer, predict, train
from smi.prompting.retrieve import POOLINGS, embed, example_pool, nearest_per_class, random_per_class, write_shots_file
from trial_run import SETTINGS as C1_SETTINGS

# C1's encoder training (scripts/trial_run.py) with the lr and epochs C1 adopted (C1 exp02 , exp03)
ENCODER = {**C1_SETTINGS, "lr": 1e-5, "epochs": 3}


def loader(split_dir: Path, tokenizer, shuffle: bool, seed: int = 0):
    """C1's input pipeline for one split (k = 0 , markers , separator , max_length as the encoder was trained)."""
    return build_dataloader(
        split_dir, tokenizer, past_k=ENCODER["past_k"], future_k=ENCODER["future_k"],
        batch_size=ENCODER["batch_size"], chunk_batches=ENCODER["chunk_batches"], shuffle=shuffle,
        max_length=ENCODER["max_length"], separator=ENCODER["separator"], markers=tuple(ENCODER["markers"]), seed=seed,
    )


def fine_tune(model, tokenizer, data_root: Path, seed: int, out: Path) -> None:
    """C1's training loop (smi.models.model.train) on train ; the model keeps its last epoch's weights."""
    cuda = torch.cuda.is_available()
    # on its device before the optimizer is built : fused AdamW needs the parameters on the GPU
    model.to(torch.device("cuda" if cuda else "cpu"))
    train_loader = loader(data_root / "train", tokenizer, shuffle=True, seed=seed)
    val_loader = loader(data_root / "val", tokenizer, shuffle=False)
    optimizer = build_optimizer(model, ENCODER["lr"], ENCODER["weight_decay"], fused=ENCODER["fused_optimizer"] and cuda)
    scheduler = build_scheduler(optimizer, train_loader, ENCODER["epochs"], ENCODER["warmup_ratio"], ENCODER["grad_accum_steps"])

    with open(out / "history.jsonl", "x", encoding="utf-8") as history:
        def on_log(line: dict) -> None:
            history.write(json.dumps(line) + "\n")
            history.flush()
            if line["type"] == "epoch":
                print(f"encoder epoch {line['epoch']}: val macro-F1 {line['val_macro_f1']:.4f} "
                      f"({line['seconds'] / 60:.1f} min)", flush=True)

        train(model, train_loader, val_loader, load_gold(data_root / "val"), optimizer, scheduler, ENCODER["epochs"],
              ENCODER["grad_accum_steps"], ENCODER["max_grad_norm"], ENCODER["log_every"], on_log=on_log,
              fp16=ENCODER["fp16"] and cuda)


def grade(records: dict, out: Path, split_dir: Path, settings: dict) -> None:
    """write records as predictions in out/ and grade them with the harness."""
    out.mkdir()
    write_predictions(records, out / "predictions.jsonl")
    result = evaluate(out / "predictions.jsonl", split_dir, out / "metrics.json", extra={"retrieval": settings})
    per_class = {label: round(result["per_class"][label]["f1"], 3) for label in LABELS}
    print(f"{out.name}: macro-F1 (6 targets) = {result['primary']['macro_f1_6']:.4f}\n  per-class F1 = {per_class}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="folder with train/ val/ source.json , e.g. data/v3_meetings")
    parser.add_argument("--out", required=True, help="the shots file's folder (must not exist)")
    parser.add_argument("--select", required=True, choices=["similar", "random"])
    parser.add_argument("--per-class", type=int, default=2, help="examples of each class per segment")
    parser.add_argument("--seed", type=int, default=0, help="random : the draw ; similar --fine-tune : the training seed")
    parser.add_argument("--encoder", default=None, help="similar : Hub name or local folder , e.g. microsoft/deberta-v3-base")
    parser.add_argument("--fine-tune", action="store_true", help="similar : fine-tune the encoder on train first (C1)")
    parser.add_argument("--pooling", default=None, choices=POOLINGS, help="similar : cls (fine-tuned) or mean")
    parser.add_argument("--epochs", type=int, default=None, help="override C1's 3 epochs (smoke tests)")
    args = parser.parse_args(argv)
    if args.select == "similar" and (args.encoder is None or args.pooling is None):
        parser.error("--select similar needs --encoder and --pooling")
    if args.select == "random" and (args.encoder is not None or args.fine_tune or args.pooling is not None):
        parser.error("--select random uses no encoder")
    if args.epochs is not None:
        ENCODER["epochs"] = args.epochs

    data_root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True)  # fails if it exists : outputs are never overwritten
    source = json.loads((data_root / "source.json").read_text(encoding="utf-8"))
    train_set = SegmentDataset(data_root / "train", ENCODER["past_k"], ENCODER["future_k"])
    val_set = SegmentDataset(data_root / "val", ENCODER["past_k"], ENCODER["future_k"])
    pool = example_pool(train_set)
    pool_labels = [train_set[i]["label"] for i in pool]
    val_ids = [val_set[i]["seg_id"] for i in range(len(val_set))]
    counts = Counter(LABELS[label] for label in pool_labels)
    print(f"{len(train_set)} train segments , {len(pool)} distinct examples : {dict(sorted(counts.items()))}", flush=True)

    settings = {
        "select": args.select, "per_class": args.per_class, "seed": args.seed, "data_version": source["data_version"],
        "split": "val", "past_k": ENCODER["past_k"], "future_k": ENCODER["future_k"],
        "pool_size": len(pool), "pool_per_class": {label: counts[label] for label in LABELS},
    }
    similarity = None
    if args.select == "random":
        rows = random_per_class(len(val_ids), pool_labels, args.per_class, args.seed)
    else:
        set_seed(args.seed)
        model, tokenizer = load_model_and_tokenizer(args.encoder)  # its new classifier head is unused without --fine-tune
        if args.fine_tune:
            fine_tune(model, tokenizer, data_root, args.seed, out)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        fp16 = ENCODER["fp16"] and device.type == "cuda"
        val_loader = loader(data_root / "val", tokenizer, shuffle=False)
        vectors = embed(model, loader(data_root / "train", tokenizer, shuffle=False), args.pooling, fp16)
        vectors.update(embed(model, val_loader, args.pooling, fp16))
        encoder_input = {k: ENCODER[k] for k in ("max_length", "separator", "markers", "batch_size", "fp16")}
        settings.update(encoder=args.encoder, fine_tune=args.fine_tune, pooling=args.pooling,
                        encoder_settings=dict(ENCODER) if args.fine_tune else encoder_input)
        if args.fine_tune:
            grade(predict(model, val_loader, fp16=fp16), out / "encoder", data_root / "val", settings)

        pool_matrix = torch.stack([vectors[train_set[i]["seg_id"]] for i in pool]).to(device)
        queries = torch.stack([vectors[seg_id] for seg_id in val_ids]).to(device)
        rows, sims = nearest_per_class(queries, pool_matrix, pool_labels, args.per_class)
        similarity = dict(zip(val_ids, sims))

    shots = {seg_id: [train_set[pool[row]]["seg_id"] for row in rows[q]] for q, seg_id in enumerate(val_ids)}
    write_shots_file(out, settings, shots, similarity)
    print(f"wrote {out}", flush=True)

    if similarity is not None:
        # the retriever alone : the class whose closest example is the closest (the first of each class is its closest)
        records = {}
        for seg_id, sims in similarity.items():
            firsts = sims[:: args.per_class]
            records[seg_id] = {"label": LABELS[firsts.index(max(firsts))], "scores": None}
        grade(records, out / "nearest", data_root / "val", settings)


if __name__ == "__main__":
    main()
