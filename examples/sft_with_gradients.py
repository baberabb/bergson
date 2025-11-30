"""
Simple SFT script that collects gradients for data attribution.

Usage:
    python examples/sft_with_gradients.py --model EleutherAI/pythia-160m --dataset NeelNanda/pile-10k
"""

from pathlib import Path

import click
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)


@click.command()
@click.option("--model", default="EleutherAI/pythia-160m", help="HuggingFace model")
@click.option(
    "--dataset",
    "dataset_name",
    default="baber/hh-rlhf-combined",
    help="HuggingFace dataset",
)
@click.option("--split", default="upsample_neg_replacement", help="Dataset split")
@click.option("--text_column", default="prompt", help="Text column name")
@click.option("--output_dir", default="runs/sft_grads", help="Output directory")
@click.option("--max_length", default=1024, help="Max sequence length")
@click.option("--batch_size", default=4, help="Batch size")
@click.option("--grad_accum", default=4, help="Gradient accumulation steps")
@click.option("--lr", default=2e-5, help="Learning rate")
@click.option("--epochs", default=1, help="Number of epochs")
@click.option("--projection_dim", default=16, help="Gradient projection dim")
@click.option("--use_lora", is_flag=True, help="Use LoRA")
@click.option("--lora_r", default=32, help="LoRA rank")
@click.option("--seed", default=1234, help="Random seed")
@click.option("--no_grads", is_flag=True, help="Disable gradient collection")
def main(
    model: str,
    dataset_name: str,
    split: str,
    text_column: str,
    output_dir: str,
    max_length: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    epochs: int,
    projection_dim: int,
    use_lora: bool,
    lora_r: int,
    seed: int,
    no_grads: bool,
):
    torch.manual_seed(seed)

    print(f"Loading model: {model}")
    lm = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split=split)
    assert isinstance(ds, Dataset)

    splits = ds.train_test_split(test_size=0.05, seed=seed)
    train_ds, eval_ds = splits["train"], splits["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Gradient collection callback
    callbacks = []
    collect_grads = not no_grads
    if collect_grads:
        grad_path = Path(output_dir) / "gradients"
        print(f"Gradients will be saved to: {grad_path}")
        callbacks.append(
            GradientCollectorCallback(
                path=grad_path,
                accumulate_grads=True,
                projection_dim=projection_dim,
            )
        )

    # LoRA (optional)
    peft_config = None
    if use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r,
            target_modules="all-linear",
            lora_dropout=0,
            bias="none",
        )

    trainer = SFTTrainer(
        model=lm,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=output_dir,
            dataset_text_field=text_column,
            max_length=max_length,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            num_train_epochs=epochs,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            seed=seed,
        ),
    )

    if collect_grads:
        trainer = prepare_for_gradient_collection(trainer)

    print("Training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nDone! Model: {output_dir}")
    if collect_grads:
        print(f"Gradients: {Path(output_dir) / 'gradients' / 'train'}")


if __name__ == "__main__":
    main()
