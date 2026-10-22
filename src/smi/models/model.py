import math
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler,
)

from smi.eval.layer1_metrics import prf_from_sets
from smi.labels import LABELS, TARGET_LABELS


def load_model_and_tokenizer(
    name_or_path: str | Path,
    labels: tuple[str, ...] = LABELS,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    load a pretrained checkpoint as a sequence classifier over `labels` , and its tokenizer.
    name_or_path : a Hub name (e.g. "microsoft/deberta-v3-base") , downloaded into the HF cache if missing ,
                   or a local folder with config + tokenizer files + weights (e.g. an attached Kaggle dataset)
    labels       : the classes of the head , in order ; class i of the model = labels[i]
                   (default : the 7 LABELS , the same index the dataset uses ; the hierarchical stages use others)

    model and tokenizer come from the same name_or_path , so they can never mismatch.
    the classification head is new and randomly initialised (HF warns about it : expected).
    weights are loaded in fp32 : transformers 5 defaults to dtype="auto" (the checkpoint's own dtype ,
    fp16 for some checkpoints).
    encoder only for now : a decoder also needs a pad token set on both tokenizer and model (open , Part 5).
    """
    id2label = {}
    label2id = {}
    for i in range(len(labels)):
        id2label[i] = labels[i]
        label2id[labels[i]] = i

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        dtype=torch.float32,
    )
    return model, tokenizer


def predict(
    model: PreTrainedModel,
    dataloader: DataLoader,
    labels: tuple[str, ...] = LABELS,
) -> dict[str, dict]:
    """
    run the model over every batch of dataloader and return one record per segment , in the harness shape :
    {seg_id: {"label": the class with the highest score , "scores": {label: probability} for every label}}
    (with the 7 LABELS : the same shape load_predictions returns , so per_class_metrics can use it directly).
    labels : the model's classes , in head order (as given to load_model_and_tokenizer)

    model.eval() is set here (dropout off) ; switching back to train mode is the training loop's job.
    batches are moved to the model's device ; seg_ids stay a python list.
    softmax is done in fp32 , so the scores sum to 1 within the harness tolerance even if logits are fp16.
    records are keyed by seg_id : the eval order (longest first) does not matter.
    """
    model.eval()
    device = next(model.parameters()).device

    records = {}
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probabilities = logits.float().softmax(dim=-1).cpu().tolist()

            for i in range(len(batch["seg_ids"])):
                row = probabilities[i]

                scores = {}
                best = 0
                for j in range(len(labels)):
                    scores[labels[j]] = row[j]
                    if row[j] > row[best]:
                        best = j

                records[batch["seg_ids"][i]] = {"label": labels[best], "scores": scores}
    return records


def macro_f1_over(records: dict[str, dict], gold: dict[str, str], select_labels: tuple[str, ...]) -> tuple[float, dict]:
    """
    the epoch-selection metric : mean F1 over select_labels , computed only on the seg_ids in gold
    (records may cover more segments , e.g. stage 2 predicts every val segment but is scored on non-"other" ones).
    uses the harness's prf_from_sets , so with the 7 LABELS and select_labels = TARGET_LABELS
    it is exactly macro_f1_6 of compute_metrics.
    returns (macro F1 , {label: F1} for every label that appears in gold or predictions)
    raises ValueError if a gold seg_id has no record
    """
    missing = []
    for seg_id in gold:
        if seg_id not in records:
            missing.append(seg_id)
    if missing:
        raise ValueError(f"{len(missing)} gold seg_ids have no prediction, e.g. {sorted(missing)[:5]}")

    pred_ids = {}
    gold_ids = {}
    for seg_id in gold:
        predicted = records[seg_id]["label"]
        if predicted not in pred_ids:
            pred_ids[predicted] = set()
        pred_ids[predicted].add(seg_id)
        if gold[seg_id] not in gold_ids:
            gold_ids[gold[seg_id]] = set()
        gold_ids[gold[seg_id]].add(seg_id)

    per_class_f1 = {}
    for label in sorted(set(pred_ids) | set(gold_ids) | set(select_labels)):
        per_class_f1[label] = prf_from_sets(pred_ids.get(label, set()), gold_ids.get(label, set()))["f1"]

    # sum() like the harness's macro_f1_6 (python's sum adds floats with extra precision) : bit-identical results
    values = [per_class_f1[label] for label in select_labels]
    return sum(values) / len(select_labels), per_class_f1


def classification_report(
    records: dict[str, dict],
    gold: dict[str, str],
    labels: tuple[str, ...],
    select_labels: tuple[str, ...],
) -> dict:
    """
    precision / recall / F1 / support per class and a confusion matrix , for any label set
    (e.g. one stage of the hierarchical model) ; scored only on the seg_ids in gold , like macro_f1_over.
    labels        : every class , in the order of the confusion matrix
    select_labels : the classes macro_f1 is averaged over
    returns {"n_segments" , "macro_f1" , "select_labels" , "per_class": {label: {"p" , "r" , "f1" , "support"}} ,
             "confusion": {"labels" , "counts"}}  (rows = gold , columns = predicted)
    raises ValueError if a gold seg_id has no record , or a label is not in labels
    """
    pred_ids = {}
    gold_ids = {}
    for label in labels:
        pred_ids[label] = set()
        gold_ids[label] = set()

    counts = []
    for i in range(len(labels)):
        counts.append([0] * len(labels))

    for seg_id in gold:
        if seg_id not in records:
            raise ValueError(f"gold seg_id {seg_id!r} has no prediction")
        predicted = records[seg_id]["label"]
        true = gold[seg_id]
        if predicted not in labels or true not in labels:
            raise ValueError(f"seg_id {seg_id!r}: label {predicted!r} or {true!r} not in {labels}")
        pred_ids[predicted].add(seg_id)
        gold_ids[true].add(seg_id)
        counts[labels.index(true)][labels.index(predicted)] += 1

    per_class = {}
    for label in labels:
        per_class[label] = prf_from_sets(pred_ids[label], gold_ids[label])

    # sum() like the harness's macro_f1_6 (python's sum adds floats with extra precision) : bit-identical results
    values = [per_class[label]["f1"] for label in select_labels]

    return {
        "n_segments": len(gold),
        "macro_f1": sum(values) / len(select_labels),
        "select_labels": list(select_labels),
        "per_class": per_class,
        "confusion": {"labels": list(labels), "counts": counts},
    }


def build_optimizer(model: PreTrainedModel, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """
    AdamW over the model's trainable parameters , in two groups :
    - 2-D parameters (weight matrices , embeddings) : weight_decay
    - 1-D parameters (biases , LayerNorm / RMSNorm weights) : no weight decay
    the split is by shape , not by name , so it works for every architecture (encoder and decoder norms are named differently).
    betas and eps are PyTorch's defaults.
    frozen parameters (requires_grad False , e.g. the base model under LoRA later) are left out.
    """
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2:
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    epochs: int,
    warmup_ratio: float,
    grad_accum_steps: int,
    scheduler_name: str = "linear",
) -> LRScheduler:
    """
    LR schedule over the whole training run , counted in optimizer steps (not batches) :
    total steps  = epochs * ceil(len(train_loader) / grad_accum_steps)
    warmup steps = int(warmup_ratio * total steps)
    "linear" = LR goes 0 -> lr during warmup , then lr -> 0 linearly until the last step.
    scheduler_name is any name transformers.get_scheduler knows (e.g. "linear" , "cosine").

    the training loop must call scheduler.step() after every optimizer step ,
    and also step the optimizer on the last , incomplete accumulation group of an epoch (that is what ceil counts).
    raises ValueError on epochs or grad_accum_steps < 1 , or warmup_ratio outside [0, 1)
    """
    if epochs < 1 or grad_accum_steps < 1:
        raise ValueError(f"epochs and grad_accum_steps must be >= 1, got {epochs} and {grad_accum_steps}")
    if warmup_ratio < 0 or warmup_ratio >= 1:
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")

    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(warmup_ratio * total_steps)
    return get_scheduler(
        scheduler_name,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def train(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_gold: dict[str, str],
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    epochs: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    log_every: int,
    labels: tuple[str, ...] = LABELS,
    select_labels: tuple[str, ...] = TARGET_LABELS,
    label_map: list[int] | None = None,
    on_log: Callable[[dict], None] | None = None,
) -> tuple[int, dict[str, dict], list[dict]]:
    """
    fine-tune model on train_loader , score val_loader after every epoch , keep the best epoch.
    model is moved to the GPU if there is one , else it stays on the CPU (fp32 , one device).
    labels        : the model's classes , in head order
    select_labels : best epoch = highest mean val F1 over these (default : the 6 target classes = macro_f1_6)
    label_map     : turns the dataset's 7-class label index into the model's class index
                    (label_map[i] = class index for LABELS[i] , or -100 to ignore) ; None = the 7 LABELS as they are
    val_gold      : seg_id -> gold label in the model's label space ; only these seg_ids are scored
    loss          : plain cross-entropy (no class weights ; handling minority classes is a later task)
    accumulation  : gradients of grad_accum_steps batches are summed before one optimizer step ;
                    the last , incomplete group of an epoch also gets a step (as build_scheduler counts)
    clipping      : the gradient norm is clipped to max_grad_norm before every optimizer step
    selection     : a tie keeps the earlier epoch
    on_log        : called with every history line the moment it is created (e.g. to print it and append it
                    to history.jsonl) , so progress is visible during training and kept if the run is stopped
    weights are not saved : after train the model holds the LAST epoch's weights ,
    the best epoch's val predictions are returned instead.

    returns (selected_epoch , records of the selected epoch on val , history) ; epochs are numbered from 1.
    history = list of dicts , in order :
      {"type": "step" , "step" , "epoch" , "train_loss" (mean since the last step line) , "lr" , "grad_norm" ,
       "seconds" (since the start of the epoch)}  every log_every optimizer steps
      {"type": "epoch" , "epoch" , "train_loss" (mean over the epoch) , "val_macro_f1" (over select_labels) ,
       "per_class_f1" , "is_best" , "seconds"}  after every epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    label_map_tensor = None
    if label_map is not None:
        label_map_tensor = torch.tensor(label_map, dtype=torch.long, device=device)

    history = []
    selected_epoch = 0
    best_f1 = -1.0
    best_records = {}
    step = 0
    num_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        epoch_loss_sum = 0.0
        log_loss_sum = 0.0
        log_batches = 0

        for batch_index, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            if label_map_tensor is not None:
                batch_labels = label_map_tensor[batch_labels]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_function(logits, batch_labels)

            # divide by the size of this accumulation group , so a step sees the mean loss of its batches
            group_start = (batch_index // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, num_batches - group_start)
            (loss / group_size).backward()

            epoch_loss_sum = epoch_loss_sum + loss.item()
            log_loss_sum = log_loss_sum + loss.item()
            log_batches = log_batches + 1

            is_group_end = (batch_index + 1) % grad_accum_steps == 0 or batch_index == num_batches - 1
            if is_group_end:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                lr = scheduler.get_last_lr()[0]  # the lr this step uses
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step = step + 1

                if step % log_every == 0:
                    line = {
                        "type": "step",
                        "step": step,
                        "epoch": epoch,
                        "train_loss": log_loss_sum / log_batches,
                        "lr": lr,
                        "grad_norm": grad_norm.item(),
                        "seconds": time.perf_counter() - start_time,
                    }
                    history.append(line)
                    if on_log is not None:
                        on_log(line)
                    log_loss_sum = 0.0
                    log_batches = 0

        records = predict(model, val_loader, labels=labels)
        val_macro_f1, per_class_f1 = macro_f1_over(records, val_gold, select_labels)

        # strictly greater : a tie keeps the earlier epoch
        is_best = val_macro_f1 > best_f1
        if is_best:
            best_f1 = val_macro_f1
            selected_epoch = epoch
            best_records = records

        line = {
            "type": "epoch",
            "epoch": epoch,
            "train_loss": epoch_loss_sum / num_batches,
            "val_macro_f1": val_macro_f1,
            "per_class_f1": per_class_f1,
            "is_best": is_best,
            "seconds": time.perf_counter() - start_time,
        }
        history.append(line)
        if on_log is not None:
            on_log(line)

    return selected_epoch, best_records, history
