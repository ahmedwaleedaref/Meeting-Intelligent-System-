"""
a local chat model (e.g. Qwen/Qwen2.5-0.5B-Instruct) that answers prompts with one letter.

no text is generated : the model reads the prompt once , and we read the probability it gives to each
of the 7 answer letters as the very next token. that gives every segment a label and 7 real scores ,
and the same prompt always gives the same numbers.
"""

import copy

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
    texts = [chat_text(tokenizer, p) for p in prompts]
    # the chat text already contains the special tokens , so the tokenizer must not add more
    batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

    # positions count real tokens only (0 , 1 , 2 , ...) , so a padded prompt is read exactly as if it were alone
    position_ids = (batch["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)

    with torch.inference_mode():  # no gradients : faster and less memory
        # logits_to_keep=1 : scores for the last position only (where the answer goes) , not for every
        # position , which would need a [batch , length , 151,936] tensor
        out = model(**batch, position_ids=position_ids, logits_to_keep=1)
    return letter_scores(out.logits[:, -1, :], letter_ids)


def chat_text(tokenizer, prompt: str) -> str:
    """the prompt wrapped in the model's chat format , ending where the assistant's answer starts."""
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def letter_scores(logits: torch.Tensor, letter_ids: list[int]) -> list[dict]:
    """the "probs" and "letter_mass" of score_batch , from the last position's logits [batch , vocabulary]."""
    logits = logits.float()  # fp32 for the softmax
    if not torch.isfinite(logits).all():
        raise FloatingPointError("the model produced NaN or inf scores (fp16 overflow?)")

    full = torch.softmax(logits, dim=-1)                      # probability of every possible next token
    letter_mass = full[:, letter_ids].sum(dim=-1)             # share that went to A..G
    probs = torch.softmax(logits[:, letter_ids], dim=-1)      # the 7 letters only , summing to 1

    return [{"probs": probs[i].tolist(), "letter_mass": letter_mass[i].item()} for i in range(len(logits))]


# ------------------------------------------------------------------------------
# prefix cache : the tokens every prompt of a run shares are run once
# ------------------------------------------------------------------------------
# every prompt of a run starts with the same instruction , options and few-shot examples ; only the
# transcript and the closing line differ. the model's keys and values for those shared tokens do not depend
# on what follows (causal attention) , so they are computed once and reused by every batch.


def tokenize_prompts(tokenizer, prompts: list[str]) -> list[list[int]]:
    """the token ids of each prompt in the model's chat format (as score_batch reads it , without padding)."""
    return tokenizer([chat_text(tokenizer, p) for p in prompts], add_special_tokens=False)["input_ids"]


def common_prefix_len(sequences: list[list[int]]) -> int:
    """
    how many leading token ids all sequences share , leaving at least one token of each sequence outside :
    the answer is read after the last token , which must go through the model.
    comparing token ids (not text) makes prefix + suffix exactly the tokens of the full prompt.
    """
    shortest = min(len(s) for s in sequences)
    first = sequences[0]
    for i in range(shortest - 1):
        if any(s[i] != first[i] for s in sequences):
            return i
    return shortest - 1


def build_prefix_cache(model, prefix_ids: list[int], batch_size: int):
    """the model's key / value cache for prefix_ids , one copy per row of a batch of batch_size prompts."""
    ids = torch.tensor([prefix_ids] * batch_size, device=model.device)
    with torch.inference_mode():
        return model(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values


def score_suffixes(model, letter_ids: list[int], prefix_cache, prefix_len: int, suffixes: list[list[int]],
                   pad_id: int) -> list[dict]:
    """
    score_batch for prompts made of the cached prefix + each suffix (token ids) : only the suffixes go through
    the model. len(suffixes) must be the batch size prefix_cache was built for.
    the suffixes are padded on the left , i.e. between the prefix and each suffix : the padding is masked out
    and the positions continue from the prefix over real tokens , so each prompt is read as if it were alone.
    """
    width = max(len(s) for s in suffixes)
    ids = torch.tensor([[pad_id] * (width - len(s)) + s for s in suffixes], device=model.device)
    suffix_mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in suffixes], device=model.device)
    prefix_mask = torch.ones(len(suffixes), prefix_len, dtype=suffix_mask.dtype, device=model.device)
    position_ids = prefix_len + (suffix_mask.cumsum(dim=-1) - 1).clamp(min=0)

    cache = copy.deepcopy(prefix_cache)  # the forward pass appends the suffixes to it : keep the prefix-only cache
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=torch.cat([prefix_mask, suffix_mask], dim=1),
                    position_ids=position_ids, past_key_values=cache, logits_to_keep=1)
    return letter_scores(out.logits[:, -1, :], letter_ids)
