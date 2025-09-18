#!/usr/bin/env python3
"""
Pretrain a two-layer transformer and try to identify the formation of induction heads
from the influence functions wrt simple induction head completions gradients.

This script:
1. Creates a 2-layer transformer using HF transformers architecture
2. Trains on TinyStories dataset using HF Trainer with Bergson callback
3. Builds a static query Bergson index using synthetic induction head data
4. Uploads the trained model to HF hub
"""

# attn_only.py
import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
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

import wandb

# from bergson.data import load_gradient_dataset
from bergson import HeadConfig
from bergson.attributor import Attributor
from bergson.collection import collect_gradients
from bergson.gradients import GradientProcessor
from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)


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


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AttnOnlyConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
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
        q = q + pos_emb
        k = k + pos_emb

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if layer_past is not None:
            pk, pv = layer_past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

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
        # self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)

    def forward(
        self,
        x,
        pos_emb,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        # self.ln_1(x)
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
        # self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
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
        x = self.wte(input_ids)  # + self.wpe(pos)
        x = self.drop(x)

        pos_emb = self.wpe(pos)
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

        # x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
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


def create_transformer():
    """Create a transformer model using GPTNeo architecture."""
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    # Alternative: use the EleutherAI 10k token tokenizer custom-built for TinyStories

    # config = GPTNeoConfig(
    #     vocab_size=len(tokenizer),
    #     hidden_size=768,
    #     intermediate_size=2,
    #     num_layers=2,
    #     num_heads=2,
    #     max_position_embeddings=1024,
    #     attention_types=[[["global"], 2]],
    #     window_size=256,
    #     resid_pdrop=0.0,
    #     embd_pdrop=0.0,
    #     attn_pdrop=0.0,
    #     layer_norm_epsilon=1e-5,
    #     initializer_range=0.02,
    #     use_cache=True,
    #     # Token IDs from the tokenizer
    #     pad_token_id=tokenizer.pad_token_id,
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    # )
    # model = GPTNeoForCausalLM(config)

    cfg = AttnOnlyConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        max_position_embeddings=1024,
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


def load_tinystories_data(tokenizer, max_length=512, N: int | None = 10_000):
    """Load and preprocess TinyStories dataset."""
    dataset = load_dataset("EleutherAI/SmolLM2-135M-10B", split="train")
    if N is not None:
        dataset = dataset.select(range(min(N, len(dataset))))
    # dataset = load_dataset("roneneldan/TinyStories", split="train")
    # dataset = dataset.select(range(min(N, len(dataset))))

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


def create_induction_head_dataset(tokenizer, seed, num_prompts=100):
    random.seed(seed)

    # crude word list, can be expanded
    base_words = [
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
        "color",
        "blue",
        "green",
        "red",
        "gold",
        "day",
        "night",
        "king",
        "queen",
        "child",
        "story",
    ]
    vocab = build_single_token_vocab(tokenizer, base_words)
    print(f"Vocab size: {len(vocab)}")

    patterns = [
        "The {A} saw the {B}. The {A}",
        "Once the {A} met the {B}, later the {A}",
        "In the story the {A} followed the {B}. The {A}",
        "My favorite is the {A} with the {B}. The {A}",
        "Everyone said the {A} remembers the {B}. The {A}",
    ]

    dataset = []
    for _ in range(num_prompts):
        try:
            A, B = random.sample(vocab, 2)
        except ValueError:
            print(f"Vocab size: {len(vocab)}")
            breakpoint()
            raise ValueError("Not enough unique tokens in vocab")

        template = random.choice(patterns)
        text = template.format(A=A, B=B)
        toks = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = toks["input_ids"][0]
        labels = torch.full_like(input_ids, -100)

        A_id = tokenizer(A, add_special_tokens=False)["input_ids"][0]
        B_id = tokenizer(B, add_special_tokens=False)["input_ids"][0]

        # mask all A and B positions
        matches_A = (input_ids == A_id).nonzero(as_tuple=True)[0]
        matches_B = (input_ids == B_id).nonzero(as_tuple=True)[0]
        labels[matches_A] = A_id
        labels[matches_B] = B_id

        # explicitly make sure final label is B
        labels[-1] = B_id

        dataset.append(
            {
                "input_ids": input_ids,
                "attention_mask": toks["attention_mask"][0],
                "labels": labels,
                "A": A,
                "B": B,
                "text": text,
            }
        )
    return dataset


def test_induction_head_labels():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
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
):
    """Set up the training configuration with Bergson callback."""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        # per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=1000,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        # save_strategy="steps",
        # save_steps=1000,
        # save_total_limit=3,
        # load_best_model_at_end=True,
        # metric_for_best_model="train_loss",
        # greater_is_better=False,
        report_to="wandb" if wandb else None,
        run_name="2-layer-transformer-smollm2-corpus",
        seed=42,
        fp16=False,
        dataloader_drop_last=True,
    )

    bergson_callback = GradientCollectorCallback(
        path=f"{output_dir}/gradients",
        head_cfgs={
            "h.0.attn.c_attn": HeadConfig(12, 192, 2),
            "h.0.attn.c_proj": HeadConfig(12, 64, 2),
            "h.1.attn.c_attn": HeadConfig(12, 192, 2),
            "h.1.attn.c_proj": HeadConfig(12, 64, 2),
        },
        projection_dim=projection_dim,
        dtype=np.float32,
        accumulate_grads=False,
        track_order=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[bergson_callback],
    )

    # Prepare for gradient collection
    trainer = prepare_for_gradient_collection(trainer)

    return trainer


def build_induction_index(
    model, induction_prompts, output_dir, projection_dim, unit_norm
):
    """Build static query Bergson index using synthetic induction head data."""
    print("Building Bergson index for induction head queries...")

    # Convert induction prompts to dataset format
    induction_data = []
    for prompt_data in induction_prompts:
        # Create a simple dataset entry
        induction_data.append(
            {
                "input_ids": prompt_data["input_ids"].tolist(),
                "attention_mask": prompt_data["attention_mask"].tolist(),
                "labels": prompt_data["input_ids"].tolist(),  # For language modeling
                "text": prompt_data["text"],
            }
        )
        # Mask out everything except the last token in the labels
        labels = [-100] * len(prompt_data["input_ids"])
        labels[-1] = prompt_data["input_ids"][-1]
        induction_data[-1]["labels"] = labels

    induction_dataset = Dataset.from_list(induction_data)

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
        path=f"{output_dir}/induction_gradients",
        skip_preconditioners=False,
    )

    # Build the attributor for querying
    print("Building attributor for querying...")
    attributor = Attributor(
        index_path=f"{output_dir}/induction_gradients",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        unit_norm=unit_norm,
    )

    # Collect mean gradient from attributor index
    mean_gradient = attributor.grads.mean(dim=0)

    attributor = Attributor(
        index_path=f"{output_dir}/induction_gradients",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        modules=True,
        unit_norm=unit_norm,
    )

    mean_module_gradients = {
        name: attributor.grads[name].mean(dim=0) for name in attributor.grads.keys()
    }

    print("In-context index built successfully! Returning mean gradient...")
    return mean_gradient, mean_module_gradients


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
    unit_norm = args.unit_norm
    tag = args.tag

    projection_dim = args.projection_dim
    seed = args.seed
    train = args.train
    plot = False

    print(
        "Starting 2-layer transformer pretraining with Bergson gradient collection..."
    )

    # Check authentication
    check_logins()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model and tokenizer
    model, tokenizer = create_transformer()

    # # Load TinyStories data
    if args.small:
        train_dataset, eval_dataset = load_tinystories_data(tokenizer, N=1000)
    else:
        train_dataset, eval_dataset = load_tinystories_data(tokenizer)

    # # Create induction head dataset
    test_induction_head_labels()
    induction_prompts = create_induction_head_dataset(
        tokenizer, seed=seed, num_prompts=10
    )

    # # Set up training
    trainer = setup_training(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir=f"examples/runs/transformer_2_layer{'_' + tag if tag else ''}",
        projection_dim=projection_dim,
        wandb=False,
    )

    if train:
        trainer.train()
        trainer.save_model(trainer.args.output_dir)
        tokenizer.save_pretrained(trainer.args.output_dir)

    if not plot:
        return

    # upload_to_hub(model, tokenizer)

    # Reload model and tokenizer
    # model = AutoModelForCausalLM.from_pretrained(trainer.args.output_dir)
    model = AttnOnlyForCausalLM.from_pretrained(trainer.args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(trainer.args.output_dir)
    model = model.to(device)

    # Build Bergson index for induction head queries
    mean_induction_gradients, module_induction_gradients = build_induction_index(
        model, induction_prompts, trainer.args.output_dir, projection_dim, unit_norm
    )
    model = model.cpu()

    # Load parquet table containing training order
    training_order = pq.read_table(
        str(Path(trainer.args.output_dir) / "gradients" / "training_order.parquet")
    ).to_pandas()

    # Plots
    os.makedirs("figures", exist_ok=True)

    # Calculate the inner products with the training gradients
    data = []
    for epoch_idx in range(trainer.args.num_train_epochs):
        # Read Bergson index from training
        attributor = Attributor(
            str(
                Path(trainer.args.output_dir)
                / "gradients"
                / "train"
                / f"epoch_{epoch_idx}"
            ),
            device=device,
            unit_norm=unit_norm,
            dtype=torch.float32,
            # faiss_cfg=FaissConfig(
        )
        inner_products = attributor.grads.float() @ mean_induction_gradients.float()
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
                        "global_step": row.global_step,
                        "index": i,
                        "score": score.item(),
                    }
                )
    data = pd.DataFrame(data)

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
    fig_name = f"figures/scores_{tag}" f'{"_norm" if unit_norm else ""}.pdf'
    plt.savefig(
        fig_name,
        format="pdf",
        bbox_inches="tight",
    )

    # Produce the same plot but split out by module (i.e. key in the grads mmap)
    data = []
    for epoch_idx in range(trainer.args.num_train_epochs):
        module_attributor = Attributor(
            index_path=f"{trainer.args.output_dir}/gradients/train/epoch_{epoch_idx}",
            device=device,
            dtype=torch.float32,
            modules=True,
            unit_norm=unit_norm,
        )
        for name, grads in module_attributor.grads.items():
            if "attention" not in name and "attn" not in name:
                print(f"Skipping {name}")
                continue
            else:
                print(f"Processing {name}")

            inner_products = grads.float() @ module_induction_gradients[name].float()
            for i, score in enumerate(inner_products.squeeze()):
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
    print(df)

    for module in df["module"].unique():
        name = module
        module_data = df[df["module"] == module]
        print(module_data)

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
            f"figures/module_scores_{tag}" f'{"_norm" if unit_norm else ""}_{name}.pdf'
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
        fig_name = (
            f'figures/sum{"_" + tag if tag else ""}'
            f'{"_norm" if unit_norm else ""}_{name}.pdf'
        )
        plt.savefig(
            fig_name,
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()

    # Can we use SVCCA to align the gradients?


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--unit_norm", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    main(args)
