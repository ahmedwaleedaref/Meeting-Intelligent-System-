"""Tests of the decoder path (task C2) and of the encoder path staying as it was.

Tiny random checkpoints are built on the fly , with a real Qwen2Tokenizer over a one-character-per-token vocabulary :
no download , no GPU. Each test catches one specific mistake , named in its comment.

Run from the repo root:   python -m pytest tests/models/test_decoder.py -v
"""

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
from transformers import (
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    Qwen2Config,
    Qwen2ForSequenceClassification,
    Qwen2Tokenizer,
)
from transformers.convert_slow_tokenizer import bytes_to_unicode

from scripts import trial_run
from smi.inputs.collator import SegmentCollator
from smi.labels import LABELS
from smi.models.model import build_optimizer, check_trainable_dtypes, count_parameters, load_model_and_tokenizer

TOY_GOLD = Path(__file__).resolve().parents[1] / "fixtures" / "toy" / "gold"

SUFFIX = trial_run.DECODER_SETTINGS["suffix"]
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA = {"r": 4, "alpha": 8, "dropout": 0.05, "target_modules": ATTENTION}

# the tiny model's sizes
HIDDEN = 64
HEADS = 4
KEY_VALUE_HEADS = 2
LAYERS = 2

# pad and eos are different tokens , like Qwen2.5-Instruct and Qwen3 : that is where a wrong rule goes silently wrong
PAD_ID = 0
EOS_ID = 1
VOCABULARY = {"<|endoftext|>": PAD_ID, "<|im_end|>": EOS_ID}
for character in bytes_to_unicode().values():
    VOCABULARY[character] = len(VOCABULARY)


def make_tokenizer(pad_token: str | None) -> Qwen2Tokenizer:
    return Qwen2Tokenizer(vocab=VOCABULARY, merges=[], eos_token="<|im_end|>", pad_token=pad_token)


def make_decoder_checkpoint(folder: Path, pad_token: str | None = "<|endoftext|>") -> Path:
    """as a real Qwen checkpoint : the config has no pad id , the tokenizer may or may not."""
    config = Qwen2Config(
        vocab_size=len(VOCABULARY), hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, num_key_value_heads=KEY_VALUE_HEADS, num_labels=len(LABELS),
        tie_word_embeddings=False,
    )
    assert config.pad_token_id is None
    Qwen2ForSequenceClassification(config).save_pretrained(folder)
    make_tokenizer(pad_token).save_pretrained(folder)
    return folder


def make_encoder_checkpoint(folder: Path) -> Path:
    config = DebertaV2Config(
        vocab_size=len(VOCABULARY), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_labels=len(LABELS),
    )
    DebertaV2ForSequenceClassification(config).save_pretrained(folder)
    make_tokenizer("<|endoftext|>").save_pretrained(folder)
    return folder


@pytest.fixture(scope="module")
def decoder_path(tmp_path_factory) -> Path:
    return make_decoder_checkpoint(tmp_path_factory.mktemp("decoder"))


@pytest.fixture(scope="module")
def encoder_path(tmp_path_factory) -> Path:
    return make_encoder_checkpoint(tmp_path_factory.mktemp("encoder"))


@pytest.fixture(scope="module")
def data_root(tmp_path_factory) -> Path:
    """the toy meetings as train and val , with the source.json evaluate reads."""
    root = tmp_path_factory.mktemp("data")
    shutil.copytree(TOY_GOLD, root / "train")
    shutil.copytree(TOY_GOLD, root / "val")
    (root / "source.json").write_text(json.dumps({"data_version": "toy"}), encoding="utf-8")
    return root


def item(seg_id: str, text: str) -> dict:
    return {"seg_id": seg_id, "label": 0, "left": [], "target_text": text, "right": []}


def padded_batch(tokenizer) -> dict:
    """inputs of different lengths : only the shorter rows are padded , so only they show a pad mistake."""
    collator = SegmentCollator(tokenizer, 64, "\n", markers=("[T] ", " [/T]"), suffix=SUFFIX)
    return collator([item("a", "so we should do it"), item("b", "yes"), item("c", "sorry yes")])


# ---------------------------------------------------------------------------
# the encoder path : what every option must leave alone
# ---------------------------------------------------------------------------

def test_defaults_load_the_encoder_as_before(encoder_path):
    model, tokenizer = load_model_and_tokenizer(encoder_path)

    assert type(model) is DebertaV2ForSequenceClassification  # not wrapped
    assert model.dtype == torch.float32
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.config.pad_token_id == tokenizer.pad_token_id
    assert not model.is_gradient_checkpointing


def test_encoder_needs_no_peft(encoder_path, monkeypatch):
    # C1's notebooks do not install peft : a top-level import would break every encoder run
    monkeypatch.setitem(sys.modules, "peft", None)
    load_model_and_tokenizer(encoder_path)


def test_empty_suffix_only_counts_truncation():
    # the encoder's rule : cut and counted , never an error
    collator = SegmentCollator(make_tokenizer("<|endoftext|>"), 4, "\n", markers=("[T] ", " [/T]"))
    collator([item("a", "so we should do it")])
    assert collator.n_truncated == 1


# ---------------------------------------------------------------------------
# the pad token : DESIGN.md Part 5 , task.md Part 1 q. 8
# ---------------------------------------------------------------------------

def test_model_gets_the_tokenizers_pad_id_not_eos(decoder_path):
    # the Qwen trap : pad_token_id = eos_token_id would be silently wrong here
    model, tokenizer = load_model_and_tokenizer(decoder_path)
    assert tokenizer.pad_token_id == PAD_ID != tokenizer.eos_token_id
    assert model.config.pad_token_id == PAD_ID


def test_tokenizer_without_pad_token_falls_back_to_eos(tmp_path):
    model, tokenizer = load_model_and_tokenizer(make_decoder_checkpoint(tmp_path, pad_token=None))
    assert tokenizer.pad_token_id == EOS_ID
    assert model.config.pad_token_id == EOS_ID


def test_pooled_position_is_each_rows_last_suffix_token(decoder_path):
    # the task's pooling test : every pad mistake , the wrong padding side , a suffix that ends differently
    model, tokenizer = load_model_and_tokenizer(decoder_path)
    model.eval()
    batch = padded_batch(tokenizer)
    input_ids = batch["input_ids"]
    last_suffix_token = tokenizer(SUFFIX)["input_ids"][-1]

    lengths = batch["attention_mask"].sum(dim=1)
    assert len(set(lengths.tolist())) == 3  # the rows really differ
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=batch["attention_mask"]).logits

        for row in range(input_ids.shape[0]):
            last = lengths[row] - 1
            assert input_ids[row, last] == last_suffix_token
            assert (input_ids[row, last + 1:] == PAD_ID).all()  # right padding
            alone = model(input_ids=input_ids[row, :last + 1].unsqueeze(0)).logits[0]
            assert torch.allclose(logits[row], alone, atol=1e-4)


def test_wrong_pad_id_is_what_the_pooling_test_catches(decoder_path):
    # without this , the test above could pass for a reason that has nothing to do with the pad id
    model, tokenizer = load_model_and_tokenizer(decoder_path)
    model.eval()
    batch = padded_batch(tokenizer)
    with torch.no_grad():
        right = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        model.config.pad_token_id = EOS_ID
        wrong = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

    assert torch.equal(right[0], wrong[0])  # the longest row is not padded : no difference
    assert not torch.allclose(right[1], wrong[1], atol=1e-4)  # silently wrong : no error , other logits


# ---------------------------------------------------------------------------
# the suffix : truncation cuts from the end , which is where the head reads
# ---------------------------------------------------------------------------

def test_truncating_a_suffix_raises():
    collator = SegmentCollator(make_tokenizer("<|endoftext|>"), 8, "\n", markers=("[T] ", " [/T]"), suffix=SUFFIX)
    with pytest.raises(ValueError, match="cut the suffix"):
        collator([item("long", "so we should do it so we should do it")])


def test_input_that_exactly_fits_is_not_an_error():
    tokenizer = make_tokenizer("<|endoftext|>")
    collator = SegmentCollator(tokenizer, 64, "\n", markers=("[T] ", " [/T]"), suffix=SUFFIX)
    length = len(tokenizer(collator.render(item("a", "yes")))["input_ids"])

    collator = SegmentCollator(tokenizer, length, "\n", markers=("[T] ", " [/T]"), suffix=SUFFIX)
    collator([item("a", "yes")])
    assert collator.n_truncated == 0


# ---------------------------------------------------------------------------
# LoRA : what trains , and in which dtype
# ---------------------------------------------------------------------------

def test_lora_trains_exactly_the_adapters_and_the_head(decoder_path):
    # catches a missing task_type (frozen head) , wrong target modules , accidental unfreezing
    model, _ = load_model_and_tokenizer(decoder_path, dtype=torch.float16, lora=LORA)

    key_value_width = HIDDEN // HEADS * KEY_VALUE_HEADS
    r = LORA["r"]
    per_layer = r * (HIDDEN + HIDDEN) + 2 * r * (HIDDEN + key_value_width) + r * (HIDDEN + HIDDEN)
    head = HIDDEN * len(LABELS)
    assert count_parameters(model)[0] == LAYERS * per_layer + head


def test_trainables_are_fp32_and_the_frozen_base_keeps_its_dtype(decoder_path):
    # PEFT leaves the head's trainable copy in fp16 : GradScaler then raises "Attempting to unscale FP16 gradients"
    model, _ = load_model_and_tokenizer(decoder_path, dtype=torch.float16, lora=LORA)
    for name, parameter in model.named_parameters():
        expected = torch.float32 if parameter.requires_grad else torch.float16
        assert parameter.dtype == expected, name


def test_all_linear_adapts_more_than_attention_and_keeps_the_head_plain(decoder_path):
    all_linear = {**LORA, "target_modules": "all-linear"}
    attention, _ = load_model_and_tokenizer(decoder_path, dtype=torch.float16, lora=LORA)
    wider, _ = load_model_and_tokenizer(decoder_path, dtype=torch.float16, lora=all_linear)

    assert count_parameters(wider)[0] > count_parameters(attention)[0]
    assert not any("score" in name and "lora_" in name for name, _ in wider.named_parameters())


def test_step_zero_is_the_pretrained_model(decoder_path):
    # B starts at 0 : a non-default init would not start from the pretrained function
    plain, tokenizer = load_model_and_tokenizer(decoder_path)
    wrapped, _ = load_model_and_tokenizer(decoder_path, lora=LORA)
    plain.score.weight.data.copy_(wrapped.base_model.model.score.modules_to_save["default"].weight.data)
    plain.eval()
    wrapped.eval()

    batch = padded_batch(tokenizer)
    with torch.no_grad():
        before = plain(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        after = wrapped(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    assert torch.equal(before, after)


def test_gradient_checkpointing_reaches_the_lowest_adapters(decoder_path):
    # the embedding is frozen : a checkpointed first layer could silently stop its adapters from learning
    model, tokenizer = load_model_and_tokenizer(decoder_path, lora=LORA, gradient_checkpointing=True)
    model.train()
    batch = padded_batch(tokenizer)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2, 3])).backward()

    lowest = [p for name, p in model.named_parameters() if "layers.0." in name and "lora_B" in name]
    assert len(lowest) == len(ATTENTION)
    assert all(parameter.grad.abs().sum() > 0 for parameter in lowest)


def test_fp16_base_trains_under_autocast_and_loss_scaling(decoder_path):
    # the whole fp16 path of train() , on the CPU : autocast , scaled loss , unscale , clip , step
    model, tokenizer = load_model_and_tokenizer(decoder_path, dtype=torch.float16, lora=LORA)
    optimizer = build_optimizer(model, 1e-3, 0.01)
    assert optimizer.param_groups[1]["params"] == []  # no 1-D trainables under LoRA : AdamW must accept that
    scaler = torch.amp.GradScaler("cpu")
    batch = padded_batch(tokenizer)
    labels = torch.tensor([1, 2, 3])

    model.train()
    losses = []
    for step in range(10):
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_fp16_base_without_autocast_fails(decoder_path):
    # why trial_run refuses --fp32 with an fp16 base
    model, tokenizer = load_model_and_tokenizer(decoder_path, dtype=torch.float16, lora=LORA)
    batch = padded_batch(tokenizer)
    with pytest.raises(RuntimeError):
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


def test_check_trainable_dtypes_stops_each_mistake(decoder_path):
    model, _ = load_model_and_tokenizer(decoder_path, dtype=torch.float16, lora=LORA)
    check_trainable_dtypes(model, torch.float16)

    with pytest.raises(ValueError, match="frozen parameter"):
        check_trainable_dtypes(model, torch.float32)

    head = [p for name, p in model.named_parameters() if "score" in name and p.requires_grad][0]
    head.data = head.data.half()
    with pytest.raises(ValueError, match="trainable parameter"):
        check_trainable_dtypes(model, torch.float16)

    head.data = head.data.float()
    head.requires_grad = False
    with pytest.raises(ValueError, match="head is frozen"):
        check_trainable_dtypes(model, None)


def test_4bit_without_lora_is_refused(decoder_path):
    with pytest.raises(ValueError, match="needs lora"):
        load_model_and_tokenizer(decoder_path, load_in_4bit=True)


# ---------------------------------------------------------------------------
# trial_run.py : flags , and whole runs through train , predict and the harness
# ---------------------------------------------------------------------------

BASE_ARGV = ["trial_run.py", "--mode", "flat", "--data-root", "d", "--model", "m", "--out-dir", "o"]


@pytest.fixture
def settings(monkeypatch) -> dict:
    """a private copy of SETTINGS : the flags write into it."""
    copy = dict(trial_run.SETTINGS)
    monkeypatch.setattr(trial_run, "SETTINGS", copy)
    return copy


def apply(monkeypatch, *flags: str) -> None:
    monkeypatch.setattr(sys, "argv", BASE_ARGV + list(flags))
    trial_run.apply_arguments(trial_run.parse_arguments())


def test_no_flags_keep_the_encoder_settings(settings, monkeypatch):
    before = dict(settings)
    apply(monkeypatch)
    assert settings == before
    assert settings["suffix"] == "" and settings["lora"] is None and settings["base_dtype"] == "fp32"


def test_decoder_flag_sets_suffix_lora_and_fp16_base(settings, monkeypatch):
    apply(monkeypatch, "--family", "decoder")
    assert settings["suffix"] == "\nLabel of [T]:"
    assert settings["lora"]["r"] == 16 and settings["lora"]["alpha"] == 32
    assert settings["base_dtype"] == "fp16"


def test_full_finetuning_is_the_decoder_without_lora(settings, monkeypatch):
    apply(monkeypatch, "--family", "decoder", "--full-finetuning")
    assert settings["lora"] is None and settings["base_dtype"] == "fp32"
    assert settings["suffix"] != ""


def test_lora_overrides_do_not_touch_the_shared_defaults(settings, monkeypatch):
    apply(monkeypatch, "--family", "decoder", "--lora-r", "64", "--lora-targets", "all-linear")
    assert settings["lora"]["r"] == 64 and settings["lora"]["target_modules"] == "all-linear"
    assert trial_run.DECODER_SETTINGS["lora"]["r"] == 16


@pytest.mark.parametrize("flags, message", [
    (["--family", "decoder", "--fp32"], "needs fp16 autocast"),
    (["--family", "decoder", "--full-finetuning", "--base-dtype", "fp16"], "cannot train"),
    (["--full-finetuning"], "needs --family decoder"),
    (["--lora-r", "64"], "need LoRA"),
    (["--family", "decoder", "--full-finetuning", "--lora-r", "64"], "need LoRA"),
])
def test_invalid_combinations_fail_at_start(settings, monkeypatch, flags, message):
    with pytest.raises(ValueError, match=message):
        apply(monkeypatch, *flags)


@pytest.mark.parametrize("flags, trains_everything", [
    (["--family", "decoder", "--base-dtype", "fp32"], False),
    (["--family", "decoder", "--full-finetuning"], True),
])
def test_decoder_run_end_to_end(settings, monkeypatch, tmp_path, data_root, decoder_path, flags, trains_everything):
    # --fp32 : the CPU has no fp16 path through train() ; test_fp16_base_trains covers it
    argv = ["trial_run.py", "--mode", "flat", "--seeds", "42", "--epochs", "1", "--fp32", "--name", "run",
            "--data-root", str(data_root), "--model", str(decoder_path), "--out-dir", str(tmp_path)] + flags
    monkeypatch.setattr(sys, "argv", argv)
    trial_run.main()

    run_dir = tmp_path / "run_seed42"
    written = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    assert written["suffix"] == SUFFIX
    assert set(written["versions"]) == {"torch", "transformers", "peft"}
    assert "commit" in written

    parameters = json.loads((run_dir / "flat" / "parameters.json").read_text(encoding="utf-8"))
    assert (parameters["trainable"] == parameters["total"]) == trains_everything
    assert 0 <= json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["primary"]["macro_f1_6"] <= 1


def test_encoder_run_end_to_end(settings, monkeypatch, tmp_path, data_root, encoder_path):
    argv = ["trial_run.py", "--mode", "flat", "--seeds", "42", "--epochs", "1", "--fp32", "--name", "run",
            "--data-root", str(data_root), "--model", str(encoder_path), "--out-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    trial_run.main()

    run_dir = tmp_path / "run_seed42"
    assert json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))["suffix"] == ""
    parameters = json.loads((run_dir / "flat" / "parameters.json").read_text(encoding="utf-8"))
    assert parameters["trainable"] == parameters["total"]
    assert (run_dir / "metrics.json").exists()
