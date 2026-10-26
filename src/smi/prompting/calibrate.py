"""
contextual calibration (Zhao et al. 2021 , "Calibrate Before Use" , https://arxiv.org/abs/2102.09690).

a prompted model prefers some answer letters whatever the input says. to measure that preference ,
ask the model about a segment with no content ("N/A" , "" , "[MASK]") : its answer is then pure bias.
dividing every real answer's probabilities by those bias probabilities (and renormalising) removes
the preference , so a letter only wins when the content pushes it above its usual level.
"""

from smi.prompting.backend import score_batch
from smi.prompting.render import render_prompt

# the content-free inputs of the paper
CONTENT_FREE = ("N/A", "", "[MASK]")


def content_free_probs(model, tokenizer, letter_ids, template, options, shots, past_k, future_k) -> list[float]:
    """
    the model's average 7 letter probabilities for content-free segments , with the same prompt
    (template , options , few-shot examples , and context lines , all content-free too).
    """
    prompts = []
    for text in CONTENT_FREE:
        item = {"seg_id": "content-free", "label": None, "target_text": text,
                "left": [text] * past_k, "right": [text] * future_k}
        prompts.append(render_prompt(item, template, options, shots))
    results = score_batch(model, tokenizer, letter_ids, prompts)
    # average over the content-free inputs , letter by letter
    return [sum(r["probs"][j] for r in results) / len(results) for j in range(len(letter_ids))]


def calibrate(probs: list[float], bias: list[float]) -> list[float]:
    """divide each probability by the letter's bias , then rescale so the 7 values sum to 1 again."""
    divided = [p / b for p, b in zip(probs, bias)]
    total = sum(divided)
    return [d / total for d in divided]
