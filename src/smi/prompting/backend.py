"""
a local chat model (e.g. Qwen/Qwen2.5-0.5B-Instruct) that answers prompts with one letter.

no text is generated : the model reads the prompt once , and we read the probability it gives to each
of the 7 answer letters as the very next token. that gives every segment a label and 7 real scores ,
and the same prompt always gives the same numbers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smi.prompting.render import LETTERS


def load_model(name: str, load_in_4bit: bool = False):
    """
    the model and its tokenizer , on the GPU if there is one.
    GPU : fp16 (the T4 has no bf16) ; CPU : fp32 (fp16 is slow and badly supported on CPU).
    load_in_4bit : store the weights in 4 bits (needs the bitsandbytes package and a GPU) , for models
                   too big for the GPU in fp16 , e.g. a 7B model on a 15 GB T4.
    returns (model , tokenizer , letter_ids) ; letter_ids = the token id of each answer letter , in LETTERS order
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    # pads on the left : every prompt in a batch then ends at the last position , where we read the answer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        from transformers import BitsAndBytesConfig  # only needed (and only installed) on the GPU machine

        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quantization, device_map={"": 0})
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()  # inference only : switches off dropout

    # each letter must be exactly one token , otherwise "the probability of the next token" is not
    # the probability of that answer
    letter_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{name}: answer letter {letter!r} is {len(ids)} tokens, expected 1")
        letter_ids.append(ids[0])
    return model, tokenizer, letter_ids


def score_batch(model, tokenizer, letter_ids: list[int], prompts: list[str]) -> list[dict]:
    """
    one forward pass over a batch of prompts.
    returns one dict per prompt , in the same order :
        "probs"     : 7 probabilities (LETTERS order) , renormalised over the 7 letters , summing to 1
        "letter_mass" : how much of the model's full next-token probability fell on the 7 letters (0..1) ;
                        low = the model wanted to say something else , so the prompt is not being followed
    """
    # wrap each prompt in the model's chat format , ending where the assistant's answer starts
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    # the chat text already contains the special tokens , so the tokenizer must not add more
    batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

    # positions count real tokens only (0 , 1 , 2 , ...) , so a padded prompt is read exactly as if it were alone
    position_ids = (batch["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)

    with torch.inference_mode():  # no gradients : faster and less memory
        # logits_to_keep=1 : scores for the last position only (where the answer goes) , not for every
        # position , which would need a [batch , length , 151,936] tensor
        out = model(**batch, position_ids=position_ids, logits_to_keep=1)
    logits = out.logits[:, -1, :].float()  # [batch , vocabulary] , in fp32 for the softmax

    if not torch.isfinite(logits).all():
        raise FloatingPointError("the model produced NaN or inf scores (fp16 overflow?)")

    full = torch.softmax(logits, dim=-1)                      # probability of every possible next token
    letter_mass = full[:, letter_ids].sum(dim=-1)             # share that went to A..G
    probs = torch.softmax(logits[:, letter_ids], dim=-1)      # the 7 letters only , summing to 1

    return [{"probs": probs[i].tolist(), "letter_mass": letter_mass[i].item()} for i in range(len(prompts))]
