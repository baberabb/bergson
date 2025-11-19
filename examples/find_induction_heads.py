#!/usr/bin/env python3
"""
Pretrain a two-layer transformer and try to identify the formation of induction heads
from the influence functions with respect to simple induction head completion gradients.

This script:
1. Creates a 2-layer attention-only transformer
2. Trains using the HF Trainer with the Bergson callback to collect gradients
3. Builds a static query Bergson index using synthetic induction head data
4. Plots the influence of the training examples on the induction heads
"""

import math
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import EvalPrediction

from bergson import (
    AttentionConfig,
    Attributor,
    FaissConfig,
    GradientProcessor,
    IndexConfig,
    collect_gradients,
)
from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)
from bergson.utils import assert_type

HEAD_CFGS = {
    "h.0.attn.c_attn": AttentionConfig(12, 192, 2),
    "h.0.attn.c_proj": AttentionConfig(12, 64, 2),
    "h.1.attn.c_attn": AttentionConfig(12, 192, 2),
    "h.1.attn.c_proj": AttentionConfig(12, 64, 2),
}


class AttnOnlyConfig(PretrainedConfig):
    model_type = "attn_only"

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
        layer_norm_epsilon=1e-5,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True,
        layer_norm=False,
        special_pos_embed=True,
        use_sdpa=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_epsilon = layer_norm_epsilon
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.use_cache = use_cache
        self.layer_norm = layer_norm
        self.special_pos_embed = special_pos_embed
        self.use_sdpa = use_sdpa


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AttnOnlyConfig):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.special_pos_embed = config.special_pos_embed
        self.use_sdpa = config.use_sdpa
        if not self.use_sdpa:
            self.register_buffer(
                "mask",
                torch.tril(
                    torch.ones(
                        config.max_position_embeddings, config.max_position_embeddings
                    )
                ).view(
                    1, 1, config.max_position_embeddings, config.max_position_embeddings
                ),
                persistent=False,
            )

    def _split_heads(self, x):
        B, T, C = x.shape
        x = x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return x

    def _merge_heads(self, x):
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)

    def forward(
        self,
        x,
        pos_emb,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # add position to q and k only
        if self.special_pos_embed:
            q = q + pos_emb
            k = k + pos_emb

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if layer_past is not None:
            pk, pv = layer_past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.config.attn_pdrop if self.training else 0.0,
                is_causal=True,
            )

        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal = self.mask[:, :, :T, : k.size(-2)]
            att = att.masked_fill(causal == 0, float("-inf"))
            if attn_mask is not None:
                att = att + attn_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = self._merge_heads(y)
        y = self.resid_drop(self.c_proj(y))
        present = (k, v) if use_cache else None
        return y, present


class AttnOnlyBlock(nn.Module):
    def __init__(self, config: AttnOnlyConfig):
        super().__init__()
        if config.layer_norm:
            self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.ln_1 = None
        self.attn = CausalSelfAttention(config)

    def forward(
        self,
        x,
        pos_emb,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if self.ln_1 is not None:
            x = self.ln_1(x)

        a, present = self.attn(
            x, pos_emb, layer_past=layer_past, use_cache=use_cache, attn_mask=attn_mask
        )
        x = x + a
        return x, present


class AttnOnlyForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = AttnOnlyConfig

    def __init__(self, config: AttnOnlyConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [AttnOnlyBlock(config) for _ in range(config.num_hidden_layers)]
        )
        if config.layer_norm:
            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.ln_f = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # HF helpers
    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_emb):
        self.wte = new_emb

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_lm_head):
        self.lm_head = new_lm_head

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids)

        pos_emb = self.wpe(pos)
        if not self.config.special_pos_embed:
            x = x + pos_emb

        x = self.drop(x)

        presents = []
        for i, block in enumerate(self.h):
            layer_past = None if past_key_values is None else past_key_values[i]
            x, present = block(
                x,
                pos_emb,
                layer_past=layer_past,
                use_cache=self.config.use_cache if use_cache is None else use_cache,
            )
            if present is not None:
                presents.append(present)

        if self.ln_f is not None:
            x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents if presents else None,
            hidden_states=None,
            attentions=None,
        )


AutoConfig.register("attn_only", AttnOnlyConfig)
AutoModelForCausalLM.register(AttnOnlyConfig, AttnOnlyForCausalLM)


def check_logins():
    """Check if user is logged into HF hub and wandb."""
    print("Checking authentication...")

    # Check HF hub login
    try:
        from huggingface_hub import whoami

        whoami()
        print("✓ Logged into Hugging Face Hub")
    except Exception as e:
        print("✗ Not logged into Hugging Face Hub. Please run: huggingface-cli login")
        raise e

    # Check wandb login
    try:
        wandb.login()
        print("✓ Logged into Weights & Biases")
    except Exception as e:
        print("✗ Not logged into Weights & Biases. Please run: wandb login")
        raise e


def create_transformer(special_pos_embed):
    """Create an attention-only transformer."""
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    # Alternative: use the EleutherAI 10k token tokenizer custom-built for TinyStories,
    #  but it's harder to find good single-token words

    cfg = AttnOnlyConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        max_position_embeddings=1024,
        layer_norm=False,
        special_pos_embed=special_pos_embed,
    )
    model = AttnOnlyForCausalLM(cfg)

    # AutoConfig.register("attn_only", AttnOnlyConfig)
    # AutoModelForCausalLM.register(AttnOnlyConfig, AttnOnlyForCausalLM)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model, tokenizer


def load_data(
    tokenizer,
    N: int | float | None = None,
    name="EleutherAI/SmolLM2-135M-10B",
    max_length=512,
):
    """Load and preprocess dataset."""
    split = (
        f"train[:{N}]"
        if N is not None and N.is_integer()
        else f"train[:{(int(N * 100))}%]"
        if N is not None
        else "train"
    )
    dataset = load_dataset(name, split=split)
    dataset = assert_type(Dataset, dataset)

    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )

        # For language modeling, labels are the same as input_ids
        # TODO probably remove this
        # tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
        num_proc=16,
    )

    # Split into train/eval
    train_eval = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_eval["train"]
    eval_dataset = train_eval["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def build_single_token_vocab(tokenizer, wordlist, max_words=500):
    singles = []
    for w in wordlist:
        toks = tokenizer(w, add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            singles.append(w)
        if len(singles) >= max_words:
            break
    return singles


def label_repeating_bigrams(input_ids, pad_token_id):
    """
    Label all positions where a bigram repeats for induction head evaluation.

    For sequence [A][B]...[A][B], when we see the second [A], we should predict [B].
    This function finds all such cases and creates labels accordingly.

    Args:
        input_ids: List of token IDs
        pad_token_id: Token ID to ignore (padding)

    Returns:
        labels: List where labels[i] = token to predict at position i, or -100 if not labeled
    """
    labels = [-100] * len(input_ids)

    # Find all bigrams and their positions
    bigrams = {}  # (tok_i, tok_i+1) -> [list of positions where bigram starts]
    for i in range(len(input_ids) - 1):
        # Skip padding tokens
        if input_ids[i] == pad_token_id or input_ids[i + 1] == pad_token_id:
            continue

        bigram = (input_ids[i], input_ids[i + 1])
        if bigram not in bigrams:
            bigrams[bigram] = []
        bigrams[bigram].append(i)

    # For any bigram that appears multiple times, label the repetitions
    for bigram, positions in bigrams.items():
        if len(positions) >= 2:  # Bigram repeats!
            # Label all repetitions (skip first occurrence - that's what we learn from)
            for pos in positions[1:]:
                # At position pos, we see first element of bigram
                # We should predict second element at pos+1
                tok1, tok2 = bigram
                if pos + 1 < len(input_ids):
                    labels[pos + 1] = tok2

    return labels


def create_induction_head_dataset(tokenizer, seed, num_prompts=100):
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    random.seed(seed)

    # Separate words into appropriate A and B categories for sensible bigrams
    A_words = [
        "blue",
        "green",
        "red",
        "gold",
        "happy",
        "sad",
        "big",
        "small",
        "fast",
        "slow",
        "smart",
        "kind",
        "brave",
        "wise",
        "young",
        "old",
    ]
    B_words = [
        "cat",
        "dog",
        "bird",
        "wolf",
        "bear",
        "sun",
        "moon",
        "star",
        "book",
        "tree",
        "car",
        "road",
        "sky",
        "song",
        "king",
        "queen",
        "child",
        "story",
        "house",
        "river",
        "mountain",
        "flower",
        "cloud",
    ]

    A_vocab = build_single_token_vocab(tokenizer, A_words)
    B_vocab = build_single_token_vocab(tokenizer, B_words)
    print(f"A vocab size: {len(A_vocab)}")
    print(f"B vocab size: {len(B_vocab)}")

    # Verify that all words are indeed single tokens
    print("A vocab:", A_vocab)
    print("B vocab:", B_vocab)

    patterns = [
        "The {A} {B} was happy. The {A} {B}",
        "Once the {A} {B} played, later the {A} {B}",
        "In the story the {A} {B} ran fast. The {A} {B}",
        "My favorite is the {A} {B} that sings. The {A} {B}",
        "Everyone said the {A} {B} is smart. The {A} {B}",
    ]

    dataset = []
    for _ in range(num_prompts):
        try:
            A = random.choice(A_vocab)
            B = random.choice(B_vocab)
        except ValueError:
            print(f"A vocab size: {len(A_vocab)}, B vocab size: {len(B_vocab)}")
            raise ValueError("Not enough unique tokens in vocab")

        template = random.choice(patterns)
        text = template.format(A=A, B=B)
        toks = tokenizer(
            text,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=16,
        )
        input_ids = toks["input_ids"]
        labels = label_repeating_bigrams(input_ids, tokenizer.pad_token_id)

        dataset.append(
            {
                "input_ids": input_ids,
                "attention_mask": toks["attention_mask"],
                "labels": labels,
                "text": text,
            }
        )
    return Dataset.from_list(dataset)


def test_induction_head_labels(tokenizer):
    dataset = create_induction_head_dataset(tokenizer, seed=0, num_prompts=3)

    for ex in dataset:
        input_ids = ex["input_ids"]
        labels = ex["labels"]

        A_id = tokenizer(ex["A"], add_special_tokens=False)["input_ids"][0]
        B_id = tokenizer(ex["B"], add_special_tokens=False)["input_ids"][0]

        # check only {A, B, -100} appear
        allowed = {A_id, B_id, -100}
        assert set(labels.tolist()).issubset(allowed)

        # every A in input_ids must be in labels
        for pos in (input_ids == A_id).nonzero(as_tuple=True)[0]:
            assert labels[pos] == A_id

        # every B in input_ids must be in labels
        for pos in (input_ids == B_id).nonzero(as_tuple=True)[0]:
            assert labels[pos] == B_id

        # final token must be B
        assert labels[-1].item() == B_id


def setup_training(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    projection_dim: int,
    wandb: bool = True,
    num_train_epochs: int = 1,
):
    """Set up the training configuration with Bergson callback."""

    pad_id = -100

    def compute_metrics(eval_preds: EvalPrediction):
        # predictions: (B, T, V)
        # label_ids: with your collator, this equals input_ids: (B, T)
        preds = eval_preds.predictions[:, :-1]  # type: ignore
        labels = eval_preds.label_ids[:, 1:]  # type: ignore

        mask = labels != pad_id
        if not mask.any():  # type: ignore
            return {"accuracy": 0.0}

        acc = (preds.argmax(-1)[mask] == labels[mask]).mean()  # type: ignore
        return {"accuracy": float(acc)}

    # def compute_metrics(eval_preds):
    #     print("compute_metrics")
    #     # predictions: (B, T, V)
    #     preds = eval_preds.predictions
    #     label_ids = eval_preds.label_ids

    #     correct = 0
    #     total = 0

    #     # how many examples to print
    #     max_print = 5
    #     printed = 0

    #     for i in range(label_ids.shape[0]):
    #         seq = label_ids[i]
    #         # last non-pad index j
    #         non_pad = np.where(seq != pad_id)[0]
    #         if len(non_pad) == 0:
    #             continue
    #         j = non_pad[-1]
    #         if j == 0:
    #             continue

    #         # predicted token at position j-1 (predicting token j)
    #         pred_logits = preds[i, j - 1]
    #         pred_tok = pred_logits.argmax(-1)
    #         tgt_tok = seq[j]

    #         correct += int(pred_tok == tgt_tok)
    #         total += 1

    #         # Trigger additional info approximately 1% of the time
    #         if random.random() < 0.01:
    #             if printed < max_print:
    #                 seq_str = tokenizer.decode(seq[:j + 1], skip_special_tokens=True)
    #                 pred_str = tokenizer.decode([pred_tok])
    #                 tgt_str = tokenizer.decode([tgt_tok])
    #                 print("=" * 40)
    #                 print(f"Example {i}")
    #                 print(f"Context up to target: {seq_str}")
    #                 print(f"Target token id: {tgt_tok} ({tgt_str})")
    #                 print(f"Predicted token id: {pred_tok} ({pred_str})")
    #                 print(f"Match? {pred_tok == tgt_tok}")
    #                 printed += 1

    #     acc = correct / total

    #     return {"accuracy": acc}

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=1,
        warmup_steps=1000,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=10_000,
        # save_total_limit=3,
        report_to="wandb" if wandb else None,
        run_name="2-layer-transformer-SmolLM2-corpus",
        seed=42,
        fp16=True,
        dataloader_drop_last=False,
    )

    bergson_callback = GradientCollectorCallback(
        path=Path(f"{output_dir}/gradients"),
        attention_cfgs=HEAD_CFGS,
        projection_dim=projection_dim,
        dtype=np.float16,
        accumulate_grads=False,
        track_order=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[bergson_callback],
        compute_metrics=compute_metrics,
    )

    # Prepare for gradient collection
    trainer = prepare_for_gradient_collection(trainer)

    return trainer


def mean_query_gradients(
    model,
    induction_dataset,
    output_dir,
    projection_dim,
    unit_norm,
):
    """Build on-disk Bergson index using synthetic induction head data."""
    # Create gradient processor
    processor = GradientProcessor(
        {},
        projection_dim=projection_dim,
        reshape_to_square=False,
    )

    # Collect gradients for the induction head dataset
    print("Collecting gradients for induction head dataset...")
    collect_gradients(
        model=model,
        data=induction_dataset,
        processor=processor,
        cfg=IndexConfig(
            run_path=str(Path(output_dir) / "induction_gradients"),
            skip_preconditioners=True,
            skip_index=False,
        ),
        attention_cfgs=HEAD_CFGS,
    )

    # Build the attributor for querying
    print("Building attributor for querying...")
    attributor = Attributor(
        index_path=Path(output_dir) / "induction_gradients",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        unit_norm=unit_norm,
    )

    # Collect mean gradient from attributor index
    mean_module_gradients = {
        name: attributor.grads[name].mean(dim=0, keepdim=True)
        for name in attributor.grads.keys()
    }

    print("In-context index built successfully! Returning mean gradients...")
    return mean_module_gradients


def upload_to_hub(model, tokenizer, model_name="2layer-transformer-tinystories"):
    """Upload the trained model to Hugging Face Hub."""
    print(f"Uploading model to Hugging Face Hub as {model_name}...")

    try:
        # Push model and tokenizer
        model.push_to_hub(model_name)
        tokenizer.push_to_hub(model_name)
        print(f"✓ Successfully uploaded to https://huggingface.co/{model_name}")
    except Exception as e:
        print(f"✗ Failed to upload to HF Hub: {e}")
        raise e


def main(args):
    check_logins()

    dataset_name = "EleutherAI/SmolLM2-135M-10B"
    # dataset_name = "RonenEldan/TinyStories"
    num_train_epochs = 1

    unit_norm = args.unit_norm
    projection_dim = args.projection_dim
    seed = args.seed
    train = args.train
    analyze = args.analyze
    output_dir = args.output_dir

    # Check if output directory exists
    output_path = Path(output_dir)
    if train and output_path.exists():
        # Check if gradient files exist (indicates previous training)
        grad_path = output_path / "gradients" / "train" / "epoch_0" / "gradients.bin"
        if grad_path.exists():
            if not args.resume_from:
                raise FileExistsError(
                    f"Output directory {output_dir} already exists with training data. "
                    f"Use --resume_from to continue training, or choose a different output directory."
                )
            else:
                print(f"Resuming training from checkpoint: {output_dir}")
        else:
            print(f"Using existing output directory: {output_dir}")
    else:
        print(f"Creating new output directory: {output_dir}")

    print(
        "Starting 2-layer transformer pretraining with Bergson gradient collection..."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = create_transformer(
        special_pos_embed=not args.no_special_pos_embed
    )

    # # Create induction head dataset
    # test_induction_head_labels(tokenizer) # Outdated
    induction_dataset = create_induction_head_dataset(
        tokenizer, seed=seed, num_prompts=100
    )

    if train:
        if args.small:
            train_dataset, _ = load_data(tokenizer, name=dataset_name, N=0.4)
        else:
            train_dataset, _ = load_data(tokenizer, name=dataset_name)

        trainer = setup_training(
            model,
            tokenizer,
            train_dataset,
            eval_dataset=induction_dataset,
            output_dir=output_dir,
            projection_dim=projection_dim,
            wandb=False,
            num_train_epochs=num_train_epochs,
        )

        # Resume from checkpoint if specified
        if args.resume_from:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    if not analyze:
        return

    # upload_to_hub(model, tokenizer)

    # Get mean module gradients for induction head queries
    model = model.to(device)  # type: ignore
    mean_module_induction_gradients = mean_query_gradients(
        model,
        induction_dataset,
        output_dir,
        projection_dim,
        unit_norm,
    )
    model = model.cpu()

    # Load parquet table containing training order
    training_order_ds = assert_type(
        Dataset, load_from_disk(str(Path(output_dir) / "gradients" / "order.hf"))
    )
    training_order = assert_type(pd.DataFrame, training_order_ds.to_pandas())

    # Analyze data
    os.makedirs("figures", exist_ok=True)

    # Use output directory basename for figure names
    run_name = Path(output_dir).name

    # Calculate the mean query gradients' inner products with the training gradients
    data = []
    for epoch_idx in range(num_train_epochs):
        # Read Bergson index from training
        attributor = Attributor(
            Path(output_dir) / "gradients" / "train" / f"epoch_{epoch_idx}",
            device="cpu",
            unit_norm=unit_norm,
            dtype=torch.float16,
            faiss_cfg=FaissConfig(
                mmap_index=True, index_factory="IVF1,SQfp16", num_shards=10
            ),
        )

        # Ordered from largest to smallest like (3 2 1 ...)
        inner_products, indices = attributor.search(
            mean_module_induction_gradients, k=None
        )
        # Restore original order
        inner_products = torch.gather(inner_products, -1, indices.argsort(dim=-1))

        for i, score in enumerate(inner_products.squeeze()):
            training_metadata = training_order[
                (training_order["_idx"] == i) & (training_order["epoch"] == epoch_idx)
            ]
            if len(training_metadata) != 1:
                continue

            for row in training_metadata.itertuples(index=False):
                data.append(
                    {
                        "epoch": epoch_idx,
                        "global_step": row[
                            training_metadata.columns.get_loc("global_step")
                        ],
                        "index": i,
                        "score": score.item(),
                    }
                )
    data = pd.DataFrame(data)

    # Visualize the influence scores
    plt.figure(figsize=(12, 8))
    plt.scatter(
        data["global_step"],
        data["score"],
        alpha=0.6,
        s=20,
        # Use epoch for color
        c=data["epoch"],
    )
    plt.xlabel("Cumulative Training Steps")
    plt.ylabel("Influence Score")
    plt.title(
        f"Most Influential Training Examples "
        f"({'Normalized' if unit_norm else 'Unnormalized'})"
    )
    plt.grid(True, alpha=0.3)
    fig_name = f"figures/scores_{run_name}{'_norm' if unit_norm else ''}.pdf"
    plt.savefig(
        fig_name,
        format="pdf",
        bbox_inches="tight",
    )

    print("Module-wise scores not yet supported for FAISS index")
    exit()

    # Produce the same plot but split out by module (i.e. key in the grads mmap)
    df_path = f"figures/module_scores_{run_name}{'_norm' if unit_norm else ''}.csv"
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        print(f"Loaded module scores from {df_path}")
    else:
        data = []
        for epoch_idx in range(num_train_epochs):
            attributor = Attributor(
                index_path=f"{trainer.args.output_dir}/gradients/train/epoch_{epoch_idx}",
                device="cpu",
                unit_norm=unit_norm,
                dtype=torch.float32,
                faiss_cfg=FaissConfig(
                    mmap_index=True, index_factory="IVF1,SQfp16", num_shards=10
                ),
            )

            for name, grad in mean_module_induction_gradients.items():
                if "attention" not in name and "attn" not in name:
                    print(f"Skipping {name}")
                    continue
                else:
                    print(f"Processing {name}")

                mod_inner_products, _ = attributor.search(
                    {name: grad}, k=None, modules=[name]
                )

                for i, score in enumerate(mod_inner_products.squeeze()):
                    training_metadata = training_order[
                        (training_order["_idx"] == i)
                        & (training_order["epoch"] == epoch_idx)
                    ]
                    if len(training_metadata) != 1:
                        continue
                    for row in training_metadata.itertuples(index=False):
                        data.append(
                            {
                                "global_step": row.global_step,
                                "epoch": epoch_idx,
                                "module": name,
                                "score": score.item(),
                            }
                        )

        df = pd.DataFrame(data)
        df.to_csv(df_path, index=False)

    attn_modules = [name for name in df["module"].unique() if "attn" in name]
    non_attn_modules = [name for name in df["module"].unique() if "attn" not in name]

    for module in non_attn_modules:
        name = module
        module_data = df[df["module"] == module]

        plt.figure(figsize=(12, 8))

        plt.scatter(
            module_data["global_step"],
            module_data["score"],
            # c=module_data["epoch"],
            alpha=0.6,
            s=20,
            label=f"Module {name}",
        )
        plt.xlabel("Training Step")
        plt.ylabel("Influence Score")
        plt.title(
            f"Most Influential Training Examples for {name} "
            f"({'Normalized' if unit_norm else 'Unnormalized'})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig_name = (
            f"figures/module_scores_{run_name}{'_norm' if unit_norm else ''}_{name}.pdf"
        )
        plt.savefig(
            fig_name,
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()

        # Add a line plot with the sum of the gradients for each module
        # Sum points at each global step
        module_data = module_data.groupby(["global_step", "epoch"], as_index=False).agg(
            score=("score", "sum")
        )
        plt.figure(figsize=(12, 8))
        plt.plot(
            module_data["global_step"],
            module_data["score"],
            label=f"Module {name}",  # c=module_data["epoch"]
        )
        plt.xlabel("Training Step")
        plt.ylabel("Sum of Gradients")
        plt.title(f"Sum of Gradients for {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig_name = f"figures/sum_{run_name}{'_norm' if unit_norm else ''}_{name}.pdf"
        plt.savefig(
            fig_name,
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()

    # Plot all attention heads in one file
    n = len(attn_modules)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False, sharey=True
    )

    for ax, module in zip(axes.flatten(), attn_modules):
        module_data = df[df["module"] == module]
        ax.scatter(
            module_data["global_step"],
            module_data["score"],
            alpha=0.6,
            s=20,
        )
        ax.set_title(module)
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        f"figures/all_heads_scores_{run_name}{'_norm' if unit_norm else ''}.pdf"
    )
    plt.close(fig)

    # Single figure with each attention modules' sum-of-scores over steps
    fig, ax = plt.subplots(figsize=(6, 4))

    for module in attn_modules:
        module_data = df[df["module"] == module]
        summed = module_data.groupby("global_step")["score"].sum().reset_index()
        ax.plot(summed["global_step"], summed["score"], label=module, alpha=0.7)

    ax.set_xlabel("Step")
    ax.set_ylabel("Sum of Scores")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.legend().remove()

    plt.tight_layout()
    fig.savefig(
        f"figures/all_heads_sum_scores_{run_name}{'_norm' if unit_norm else ''}.pdf"
    )
    plt.close(fig)

    # Single figure with each attention modules' sum-of-scores summed over steps
    sums = [df[df["module"] == m]["score"].sum() for m in attn_modules]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(attn_modules)), sums)
    ax.set_xticks(range(len(attn_modules)))
    ax.set_xticklabels(attn_modules, rotation=90)
    ax.set_ylabel("Sum of Scores")
    ax.set_xlabel("Module")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        f"figures/all_heads_sum_scores_bar_{run_name}{'_norm' if unit_norm else ''}.pdf"
    )
    plt.close(fig)

    # Step 1: pick checkpoint steps
    # Step 2: compute a bunch of gradients at this step using the static index build
    #   and save it
    # Step 1.5: fix the static index build bug

    # Can we use optimal transport to align the gradients?
    # Should we transport the activations then transport the gradients in the same way?
    # Or should we transport the gradients directly?

    # To compute the optimal transport maps we just need a huge dataset of training
    # gradients at different steps.

    # Once we have optimal transport maps we can optimal transport the gradients to the
    # trained model distribution. Then we can compute the influence of the training
    # examples on the induction heads.


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for model checkpoints and gradients",
    )
    parser.add_argument("--projection_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--unit_norm", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--no_special_pos_embed", action="store_true")
    parser.add_argument(
        "--resume_from",
        action="store_true",
        help="Resume training from existing checkpoint in output_dir",
    )
    args = parser.parse_args()
    main(args)
