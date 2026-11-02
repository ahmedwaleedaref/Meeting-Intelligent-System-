# C2 Part 1 note: the decoder, understood

Answers to the nine questions of [task.md](task.md), Part 1. Where I say something was checked, it was run on the pinned versions (torch 2.14.1, transformers 5.20.1, peft 0.21.2), on the real Qwen2.5-0.5B config and tokenizer or on a tiny random Qwen. Where I give a result from our data, it's from [exp01](experiments/exp01_lora-vs-full/NOTE.md), which I ran after writing the first version of these answers.

## LoRA

### 1. What exactly is trained?

Very little. Take one weight matrix `W`, say a `q_proj` in Qwen. We freeze it completely and put a small trainable path next to it, so the layer computes `W·x + (α/r)·B·A·x`. `A` squeezes the 896 numbers of the input down to r numbers, and `B` expands them back. The only things that get trained are those `A` and `B` matrices and the new classification head.

"Low rank" is about how many directions the update can move things in. However large `ΔW = B·A` looks on paper, everything passes through a bottleneck of r numbers, so the change can only push the layer's output along r directions. For one `q_proj` at r = 16, that's 28,672 trainable numbers instead of 802,816.

The reason that can possibly be enough is that fine-tuning isn't teaching the model English. It already knows English. Fine-tuning mostly picks out and strengthens features that already exist, and the intrinsic dimensionality paper measured how little room that takes: RoBERTa reached 90 % of full fine-tuning's score with a few hundred trained numbers, out of hundreds of millions. LoRA makes the same bet, one matrix at a time.

Don't confuse this with compressing `W`. An SVD would replace `W` with a low-rank approximation and throw information away, and that would hurt, because pretrained weights really aren't low rank (Biderman et al. report that about half the singular directions are needed for 90 % of the variance). LoRA never touches `W`. It keeps everything and only restricts the *change*.

### 2. Why does B start at zero?

So that training starts from the model we paid for. With `B = 0` the extra path adds exactly nothing, every layer computes plain `W·x`, and at step 0 the model is the pretrained Qwen plus a random head. I checked it: logits before and after wrapping are exactly equal. If both matrices started random, we'd begin by adding noise to every attention layer of a good model.

You might then ask why `A` isn't zero too. If both were zero, both gradients would be zero and nothing would ever move, because `B`'s gradient depends on `A·x` and `A`'s gradient depends on `B`. With `A` random, `B` gets a real gradient on the first step, and `A` starts moving as soon as `B` isn't zero.

### 3. Why does LoRA save memory?

A parameter that trains costs far more than its value. In fp32 it needs 4 bytes for the value, 4 for its gradient and 8 for AdamW's two running averages: 16 bytes. A frozen parameter needs only its value, and since it never updates it can live in fp16 at 2 bytes. Only adapters and the head have gradients and AdamW states, so `build_optimizer` leaves everything else out on its own. For Qwen 0.5B that's the difference between about 7.9 GB and about 1 GB before activations.

What LoRA doesn't save is the activations. The gradient for an adapter in layer 3 has to travel back through layers 24 to 4, frozen or not, so the forward pass must keep its intermediate results. That's why a long batch can still run out of memory under LoRA, and why the notebook measures memory rather than trusting this arithmetic.

### 4. Choosing r and α

**r is a hyperparameter, not a law.** 8 and 16 are just common starting points. Hu et al. got away with r = 1 on some GPT-3 tasks, while Biderman et al. needed 256 for code. The right r depends on how much the task has to change the model. A 7-class head on short utterances shouldn't need much, so 16 is a sensible start, and exp01 checks whether more capacity changes anything.

**r and α have to be read together** because the adapter's output is multiplied by α/r. With AdamW each entry of `B` moves by roughly the learning rate whatever the gradient's size, so that multiplier directly sets how far the layer's output moves per step. α behaves like a second learning rate. Changing r alone quietly changes the step size as well as the capacity, which is why a result only means something when r, α and the learning rate are reported as one setting. Ours are r = 16, α = 32, so the multiplier is 2.

**Which matrices get adapters can matter more than r.** A matrix with no adapter can't change at all, whatever the rank elsewhere. Extra rank inside matrices that already have adapters tends to add directions that barely help. Hu et al., QLoRA and Biderman et al. all point the same way: coverage beats rank. In Qwen the MLP holds about 13.1 M parameters per layer against 1.8 M in attention, and attention-only LoRA leaves all of that frozen. That's why our capacity run uses all linear layers at r = 16 rather than r = 64. The two train almost the same number of parameters (8.80 M vs 8.66 M), but all-linear changes only *where* the capacity sits, while r = 64 would also force a decision about α. On our task it turned out not to matter: 0.4793 against 0.4807. That's not a counterexample to the papers. Those tasks had a gap to close, and ours didn't.

### 5. After training

Both options work. Since `W·x + (α/r)·B·A·x = (W + (α/r)·B·A)·x`, you can compute `W' = W + (α/r)·B·A` once, throw the adapters away, and have an ordinary model with no extra cost at inference. That's what `merge_and_unload()` does. It's only possible because the update is linear and sits in parallel with `W`; an adapter in series behind a non-linearity couldn't be folded in like this.

Keeping them separate has its own advantage: the saved file holds only adapters and head, about 8.7 MB against about 1 GB for the base, and several adapters can share one base model. That's what DESIGN.md Part 4 plans for checkpoints. The head must always be saved with the adapter, since it's new and random and can't be recovered from the base. `task_type="SEQ_CLS"` takes care of that.

### 6. Where LoRA falls short

LoRA falls short when the task needs a big, high-rank change: a lot of new knowledge, a large shift away from the pretraining data, and a lot of training data. Biderman et al. saw this on code and maths on Llama-2-7B, where LoRA stayed behind full fine-tuning at every rank up to 256 in continued pretraining, and the change full fine-tuning made was high rank.

Our task is much closer to the classification tasks where LoRA matched full fine-tuning: seven classes, short English segments, about 79k examples. So I expected a small gap, and exp01 found none we can see: LoRA 0.4807, full fine-tuning 0.4728, with the sign changing between seeds.

I'd still read that carefully, for two reasons. LoRA is more sensitive to the learning rate, and we didn't search it, so neither arm is necessarily at its best. And there's a trap on the other side: at k = 0 both methods may hit the same ceiling, because "yeah" alone simply doesn't say enough. The numbers look like exactly that: `aa`, `bk` and `other` score the same in every arm, and in the encoder too. A tie there doesn't prove that LoRA is equal at larger k, so it should be measured again before anyone relies on it there.

## A decoder as a classifier

### 7. Why the last real token, and what is the suffix for?

Causal attention means each token only sees itself and what came before it. So the first token knows only itself, a middle token knows a prefix, and the only position that has seen the whole input is the last one. That's the one position that can summarise the input, so that's where the head reads. Averaging all positions would mix in lots of partial views, and Label Supervised LLaMA measured how badly that goes: 55.9 % against 96.7 % on SST-2.

Without a suffix, the last token would be whatever word the input happens to end on, different in every example, and at k > 0 the target would be somewhere in the middle with no reason to look back at it. `"\nLabel of [T]:"` fixes that. Every input now ends on the same token, `]:` (I checked this on the real tokenizer), and the words give that position a job: summarise the marked segment. There's a nice side effect too. Qwen spent its pretraining learning to predict what comes next, and what comes next after "Label of [T]:" is a label. The encoder doesn't need any of this, because its `[CLS]` already sees both directions.

### 8. The pad token

In a padded batch, the classifier has to find each row's last real token. It doesn't use the attention mask for that. It looks at the token ids and takes the rightmost one that isn't `config.pad_token_id`. So the tokenizer writes the padding, and the model recognises it only by its id. That gives three ways to fail:

- **The tokenizer has no pad token:** the tokenizer raises as soon as it tries to pad. Loud, easy.
- **The model has no `pad_token_id`:** any batch above 1 raises "Cannot handle batch sizes > 1 if no padding token is defined". This is exactly the state of Qwen2.5-0.5B as it loads, so we'd hit it on the first batch.
- **The two disagree:** nothing raises, and this is the dangerous one. The model never finds the id it's looking for, so a padded row gets read at a pad position. That position still attends back to the real tokens, so its vector isn't garbage, the loss falls, and you just get a quietly worse model with nothing pointing at the cause. The longest row of each batch isn't padded and stays correct, which makes it even harder to notice. `test_wrong_pad_id_is_what_the_pooling_test_catches` shows exactly this.

The Qwen trap is an easy way into that third case. Qwen pads with `<|endoftext|>` (151643). On the base model that's also the EOS, so `pad_token_id = eos_token_id` works by coincidence. On Instruct and Qwen3 the EOS is `<|im_end|>` (151645), so the same line would be silently wrong. The rule that's right everywhere is to copy what the tokenizer actually writes: `model.config.pad_token_id = tokenizer.pad_token_id`.

### 9. Memory on a T4

**The frozen base goes in fp16** because fp16's problem doesn't apply to it. fp16 has few digits, so a small update to a weight can simply vanish. Frozen weights never get updated, so all they need is to be read, and they cost half the memory: 0.99 GB instead of 1.98 GB at 0.5B. The real risk is activations overflowing fp16's limit of 65,504, which is why the notebook's memory check asserts that the logits stay finite.

**The trainable weights stay fp32** for the opposite reason: they're the ones where small updates must survive. It's also enforced. A gradient has its parameter's dtype, and `GradScaler.unscale_` refuses fp16 gradients. PEFT upcasts `A` and `B` by itself but leaves the head's trainable copy in the base's dtype, a detail that's easy to miss, so `load_model_and_tokenizer` casts every trainable to fp32 after wrapping and then `check_trainable_dtypes` verifies it. One consequence: with an fp16 base and an fp32 head the forward fails without autocast, so `trial_run.py` refuses `--fp32` on such a run.

On the scaler: its first steps can overflow. It then skips the step, halves the scale and carries on, so a `nan` grad norm early in a run is normal. On the real model this happened more than I expected. Every run went from a scale of 65,536 to 256–512 within the first 50 steps, then climbed slowly back to 2,048–4,096. DeBERTa stayed near 65,536. It cost only 7–8 steps out of 7,398, nearly all of them while the learning rate was still tiny, so it did no harm. The full fine-tuning arm did the same with its fp32 weights, so I doubt the frozen fp16 base is the cause. My guess is that Qwen's gradients are simply larger and overflow fp16 sooner when scaled, but I haven't tested that.

**4-bit (QLoRA) becomes necessary** when even the fp16 weights don't fit: at 7B they're 14.1 GB of a T4's 15. **Gradient checkpointing becomes necessary** when activations don't fit: bigger models, longer inputs, or larger batches. C1 already found this at k = 5. It recomputes each layer's activations during backward instead of storing them, which costs about a third more time but gives the same gradients. At 0.5B and k = 0, LoRA needs neither: it peaked at 8.0 GB reserved, and notice how that compares with the 1 GB of state I calculated above. Almost all of it is activations. Full fine-tuning has about 7.9 GB of state before activations, and it really didn't fit: it ran out of memory at batch 32, and needed checkpointing to come down to 9.3 GB.
