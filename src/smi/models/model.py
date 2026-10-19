from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from smi.labels import LABELS


def load_model_and_tokenizer(name_or_path: str | Path) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    load a pretrained checkpoint as a 7-class sequence classifier , and its tokenizer.
    name_or_path : a Hub name (e.g. "microsoft/deberta-v3-base") , downloaded into the HF cache if missing ,
                   or a local folder with config + tokenizer files + weights (e.g. an attached Kaggle dataset)

    model and tokenizer come from the same name_or_path , so they can never mismatch.
    class i of the model = LABELS[i] , the same index the dataset uses for labels.
    the classification head is new and randomly initialised (HF warns about it : expected).
    weights are loaded in fp32 : transformers 5 defaults to dtype="auto" (the checkpoint's own dtype ,
    fp16 for some checkpoints) ; mixed precision is applied later by the training loop , not here.
    encoder only for now : a decoder also needs a pad token set on both tokenizer and model (open , Part 5).
    """
    id2label = {}
    label2id = {}
    for i in range(len(LABELS)):
        id2label[i] = LABELS[i]
        label2id[LABELS[i]] = i

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        name_or_path,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        dtype=torch.float32,
    )
    return model, tokenizer


def predict(model: PreTrainedModel, dataloader: DataLoader) -> dict[str, dict]:
    """
    run the model over every batch of dataloader and return one record per segment , in the harness shape :
    {seg_id: {"label": the class with the highest score , "scores": {label: probability} for the 7 LABELS}}
    (the same shape load_predictions returns , so per_class_metrics can use it directly).

    model.eval() is set here (dropout off) ; switching back to train mode is the training loop's job.
    batches are moved to the model's device ; seg_ids stay a python list.
    softmax is done in fp32 , so the 7 scores sum to 1 within the harness tolerance even if logits are fp16.
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
                for j in range(len(LABELS)):
                    scores[LABELS[j]] = row[j]
                    if row[j] > row[best]:
                        best = j

                records[batch["seg_ids"][i]] = {"label": LABELS[best], "scores": scores}
    return records
