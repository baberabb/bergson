import os
from itertools import chain
from typing import Dict, List, Optional

import datasets
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils import data
from transformers import AutoTokenizer

from bergson.approx_unrolling.logger_config import get_logger

logger = get_logger(__name__)


def get_pile_dataset(
    model_str: str = "EleutherAI/pythia-70m",
    step: int = 0,
    split: str = "train",
    block_size: int = 2048,  # Pythia typically uses 2048 context length
    indices: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
) -> data.Dataset:
    """
    Process Pile dataset for language modeling, similar to the WikiText function.

    Args:
        model_str: Pythia model to get tokenizer from
        step: Model checkpoint step
        split: Dataset split to use
        block_size: Size of each text chunk (should match model's context length)
        indices: Optional list of specific sample indices to use
        max_samples: Optional limit on number of samples to process

    Returns:
        Dataset with fixed-size chunks ready for language modeling
    """

    # Load the Pile dataset
    logger.info("Loading Pile 10k dataset...")
    raw_datasets = load_dataset("NeelNanda/pile-10k")
    assert isinstance(raw_datasets, datasets.dataset_dict.DatasetDict)
    # Load Pythia tokenizer
    logger.info(f"Loading tokenizer for {model_str} at step {step}...")
    tokenizer = AutoTokenizer.from_pretrained(model_str, revision=f"step{step}", use_fast=True, trust_remote_code=True)

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get column names
    column_names = raw_datasets["train"].column_names

    text_column_name = "text" if "text" in column_names else column_names[0]

    # Limit samples if specified
    if max_samples is not None:
        logger.info(f"Limiting to {max_samples} samples...")
        raw_datasets["train"] = raw_datasets["train"].select(range(min(max_samples, len(raw_datasets["train"]))))

    def tokenize_function(examples):
        """Tokenize the text column"""
        return tokenizer(examples[text_column_name])

    # Tokenize the dataset
    logger.info("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,  # Use multiple processes
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        """
        Concatenate all texts and split into chunks of block_size.
        This is the key function that creates uniform-sized training samples.
        """
        # Concatenate all tokenized texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Drop the last chunk if it's smaller than block_size
        total_length = (total_length // block_size) * block_size

        # Split into chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        # For language modeling, labels are the same as input_ids
        result["labels"] = result["input_ids"].copy()

        return result

    # Group texts into chunks
    logger.info(f"Grouping texts into chunks of {block_size} tokens...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    # Select the appropriate split
    if split in ["train", "eval_train"]:
        ds = lm_datasets["train"]
    else:
        # For validation, you might want to use a subset of train data
        # since NeelNanda/pile-10k only has train split
        ds = lm_datasets["train"]
        logger.info("Using train split since pile-10k only has train data")

    # Select specific indices if provided
    if indices is not None:
        logger.info(f"Selecting {len(indices)} specific samples...")
        ds = ds.select(indices)

    logger.info(f"Final dataset size: {len(ds)} samples")
    logger.info(f"Each sample has {block_size} tokens")

    return ds


def analyze_processed_dataset(dataset, tokenizer, num_samples: int = 5):
    """Analyze the processed dataset"""
    logger.info("=== Dataset Analysis ===")
    logger.info(f"Total samples: {len(dataset)}")

    # Check a few samples
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        logger.info(f"Sample {i}:")
        logger.info(f"  Input IDs shape: {len(sample['input_ids'])}")
        logger.info(f"  Labels shape: {len(sample['labels'])}")
        logger.info(f"  Are labels == input_ids? {sample['labels'] == sample['input_ids']}")

        # Decode first few tokens
        decoded_start = tokenizer.decode(sample["input_ids"][:50])
        decoded_end = tokenizer.decode(sample["input_ids"][-50:])
        logger.info(f"  Text start: {decoded_start[:100]}...")
        logger.info(f"  Text end: ...{decoded_end[-100:]}")


def create_dataloader(dataset, batch_size: int = 512):
    """Create a DataLoader"""
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader


def randomize_batch(batch: Dict[str, Tensor], rank: int):
    random_seed = int.from_bytes(os.urandom(4), "big")
    torch.manual_seed(random_seed)

    # Randomly sample how many entries to add (0 to 10)
    num_entries = torch.randint(0, 5, (1,)).item()  # 0 to 10 inclusive
    # num_entries = 4
    if num_entries == 0:
        return  # No entries to add

    for k in batch.keys():
        shape = batch[k].shape  # [b, s]
        b, s = shape[0], shape[1]

        # Create i new random entries with shape [num_entries, s]
        if batch[k].dtype == torch.long:  # For integer tensors
            random_entries = torch.randint(0, 10, (num_entries, s), device=f"cuda:{rank}")  # Adjust range as needed
        else:  # For float tensors
            random_entries = torch.randn(num_entries, s, device=f"cuda:{rank}")

        # Concatenate to get [b+i, s]
        batch[k] = torch.cat([batch[k], random_entries], dim=0)


# Example usage
if __name__ == "__main__":
    # Process the dataset
    processed_dataset = get_pile_dataset(
        model_str="EleutherAI/pythia-70m",
        step=1000,
        block_size=2048,
        max_samples=5000,  # Limit for testing
    )

    # Analyze it
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision="step143000")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    analyze_processed_dataset(processed_dataset, tokenizer)

    # Create dataloader
    dataloader = create_dataloader(processed_dataset, batch_size=512)

    # Test the dataloader
    batch = next(iter(dataloader))
    logger.info("Batch test:")
    logger.info(f"  Batch input_ids len: {len(batch['input_ids'])}")
    logger.info(f"  Batch labels len: {len(batch['labels'])}")
