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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    Trainer,
    TrainingArguments,
)

import wandb
from bergson.attributor import Attributor

# from bergson.data import load_gradient_dataset
from bergson.collection import collect_gradients
from bergson.gradients import GradientProcessor
from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)


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
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/TinyStories-restricted")

    # TODO use the EleutherAI 10k token tokenizer custom-built for TinyStories
    # Padding and truncation = True
    config = GPTNeoConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=1024,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=1024,
        attention_types=[[["global"], 2]],
        window_size=256,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        # Token IDs from the tokenizer
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPTNeoForCausalLM(config)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model, tokenizer


def load_tinystories_data(tokenizer, max_length=512, N=10000):
    """Load and preprocess TinyStories dataset."""
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(min(N, len(dataset))))

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


def create_induction_head_dataset(tokenizer, num_prompts=10):
    """Create synthetic induction head dataset for building the query index."""
    print(f"Creating {num_prompts} synthetic induction head prompts...")

    # Create induction head patterns: [A][B] ... [A] -> [B]
    # These are designed to test if the model learns to copy tokens
    # from earlier in the sequence

    # Generate diverse induction head patterns
    patterns = [
        "The cat sat on the mat. The cat",
        "Once upon a time, there was a princess. Once upon a time",
        "In the forest, the wolf howled. In the forest",
        "The sun shines bright today. The sun",
        "My favorite color is blue. My favorite color",
        "The dog ran in the park. The dog",
        "She loves to read books. She loves",
        "The moon is full tonight. The moon",
        "He plays guitar every day. He plays",
        "The bird sings a sweet song. The bird",
    ]

    # Take the requested number of prompts
    selected_prompts = patterns[:num_prompts]

    # Tokenize the prompts
    tokenized_prompts = []
    for prompt in selected_prompts:
        # Split into input and target (everything after the last space)
        parts = prompt.rsplit(" ", 1)
        if len(parts) == 2:
            input_text, target = parts
            tokenized = tokenizer(
                input_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512,
            )
            # Get the target token ID
            target_tokens = tokenizer(
                target, return_tensors="pt", add_special_tokens=False
            )
            if target_tokens["input_ids"].numel() > 0:
                target_token_id = target_tokens["input_ids"][0, 0].item()
                tokenized_prompts.append(
                    {
                        "input_ids": tokenized["input_ids"][0],
                        "attention_mask": tokenized["attention_mask"][0],
                        "target_token": target_token_id,
                        "text": prompt,
                    }
                )

    print(f"Created {len(tokenized_prompts)} induction head prompts")
    return tokenized_prompts


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
        num_train_epochs=3,
        per_device_train_batch_size=16,
        # per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=100,
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
        run_name="2-layer-transformer-tinystories",
        seed=42,
        fp16=False,
        dataloader_drop_last=True,
    )

    bergson_callback = GradientCollectorCallback(
        path=f"{output_dir}/gradients",
        projection_dim=projection_dim,
        dtype=np.float32,
        accumulate_grads=False,
        track_training_order=True,
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


def build_induction_index(model, induction_prompts, output_dir, projection_dim):
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
    )

    # Collect mean gradient from attributor index
    mean_gradient = attributor.grads.mean(dim=0)

    print("In-context index built successfully! Returning mean gradient...")
    return mean_gradient


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


def main(projection_dim=128):
    tag = ""
    k = 1000
    unit_norm = False

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
    model = model.to(device)

    # # Load TinyStories data
    train_dataset, eval_dataset = load_tinystories_data(tokenizer)

    # # Create induction head dataset
    induction_prompts = create_induction_head_dataset(tokenizer)

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

    trainer.train()

    # trainer.save_model(trainer.args.output_dir)
    # tokenizer.save_pretrained(trainer.args.output_dir)

    # upload_to_hub(model, tokenizer)

    # Reload model and tokenizer
    model = GPTNeoForCausalLM.from_pretrained(trainer.args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(trainer.args.output_dir)
    model = model.to(device)

    # Build Bergson index for induction head queries
    mean_induction_gradients = build_induction_index(
        model, induction_prompts, trainer.args.output_dir, projection_dim
    )
    model = model.cpu()

    # Read Bergson index from training
    epoch_attributors = [
        Attributor(
            str(
                Path(trainer.args.output_dir) / "gradients" / "train" / f"epoch_{epoch}"
            ),
            device=device,
            unit_norm=unit_norm,
            dtype=torch.float32,
            # faiss_cfg=FaissConfig(
        )
        for epoch in [0]  # range(trainer.args.num_train_epochs)
    ]
    # Load parquet table containing training order
    training_order = pq.read_table(
        str(Path(trainer.args.output_dir) / "gradients" / "training_order.parquet")
    ).to_pandas()

    # Test the attributor with a sample query
    print("Testing Bergson index with sample query...")
    test_prompt = "The cat sat on the mat. The cat"
    test_input = tokenizer(test_prompt, return_tensors="pt").to(device)

    # Mask out everything except the last token in the labels
    test_input["labels"] = test_input["input_ids"].clone()
    test_input["labels"][:, :-1] = -100

    top_data = []

    model = model.to(device)
    for epoch_idx, epoch_attributor in enumerate(epoch_attributors):
        # print(f"Top {k} most influential training examples for epoch {epoch_idx}:")

        with epoch_attributor.trace(model.base_model, k=k) as result:
            outputs = model(**test_input)
            outputs.loss.backward()
            model.zero_grad()

        skips = 0
        for i, (score, idx) in enumerate(
            zip(result.scores.squeeze(), result.indices.squeeze())
        ):

            if idx.item() != -1:
                # Get the training order
                training_metadata = training_order[
                    (training_order["_idx"] == idx.item())
                    & (training_order["epoch"] == epoch_idx)
                ]
                if training_metadata.empty:
                    skips += 1
                    continue
                for row in training_metadata.itertuples(index=False):
                    # print(f"{i+1}. Score: {score.item():.4f},
                    # Global step: {row.global_step}, Index: {idx.item()}")
                    top_data.append(
                        {
                            "epoch": epoch_idx,
                            "global_step": row.global_step,
                            "index": idx.item(),
                            "score": score.item(),
                        }
                    )
        print(f"Skipped {skips} examples for epoch {epoch_idx}")

    top_data = pd.DataFrame(top_data)

    # Scatter plot of scores over time
    plt.figure(figsize=(12, 8))

    for epoch in sorted(top_data["epoch"].unique()):
        epoch_data = top_data[top_data["epoch"] == epoch]
        plt.scatter(
            epoch_data["global_step"],
            epoch_data["score"],
            alpha=0.6,
            s=20,
            label=f"Epoch {epoch}",
        )

    plt.xlabel("Cumulative Training Steps")
    plt.ylabel("Influence Score")
    plt.title("Most Influential Training Examples Per Epoch (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_name = (
        f'training_dynamics{"_" + tag if tag else ""}{"_norm" if unit_norm else ""}.pdf'
    )
    plt.savefig(
        fig_name,
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()

    # Calculate the inner products with the training gradients
    data = []
    for epoch_idx, attributor in enumerate(epoch_attributors):
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
    for epoch in sorted(data["epoch"].unique()):
        epoch_data = data[data["epoch"] == epoch]
        plt.scatter(
            epoch_data["global_step"],
            epoch_data["score"],
            alpha=0.6,
            s=20,
            label=f"Epoch {epoch}",
        )

    plt.xlabel("Cumulative Training Steps")
    plt.ylabel("Influence Score")
    plt.title("Most Influential Training Examples Per Epoch (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_name = (
        f'training_dynamics_mean_induction{"_" + tag if tag else ""}'
        f'{"_norm" if unit_norm else ""}.pdf'
    )
    plt.savefig(
        fig_name,
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
