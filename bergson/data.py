import math
import os
import re
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from simple_parsing import field
from torch.utils.data import Dataset as TorchDataset

from .utils import assert_type


@dataclass
class IndexConfig:
    """Config for building the index and running the model/dataset pipeline."""

    run_path: str = field(positional=True)
    """Name of the run. Used to create a directory for the index."""

    model: str = "HuggingFaceTB/SmolLM2-135M"
    """Name of the model to load."""

    dataset: str = "EleutherAI/SmolLM2-135M-10B"
    """Dataset identifier to build the index from."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode. Requires the bitsandbytes library."""

    projection_dim: int = 16
    """Dimension of the random projection for the index, or 0 to disable it."""

    token_batch_size: int = 8192
    """Batch size in tokens for building the index."""

    prompt_column: str = "text"
    """Column in the dataset that contains the prompts."""

    completion_column: str = ""
    """Optional column in the dataset that contains the completions."""

    conversation_column: str = ""
    """Optional column in the dataset that contains the conversation."""

    normalizer: Literal["adafactor", "adam", "none"] = "adafactor"
    """Type of normalizer to use for the gradients."""

    fisher_fourth_root: bool = False
    """Whether to use the fourth root of the Fisher information matrix."""

    processor_path: str = ""
    """Path to a precomputed processor."""

    stats_sample_size: int = 10_000
    """Number of examples to use for estimating processor statistics."""

    drop_columns: bool = False
    """Only return the new dataset columns."""


class MemmapDataset(TorchDataset):
    """Torch Dataset backed by a memory-mapped numpy array."""

    def __init__(
        self,
        data_path: str,
        ctx_len: int,
        max_examples: int | None = None,
        dtype=np.uint16,
    ):
        mmap = np.memmap(data_path, dtype=dtype, mode="r").reshape(-1, ctx_len)
        self.mmap = mmap[:max_examples]

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, idx):
        return dict(input_ids=torch.from_numpy(self.mmap[idx].astype(np.int64)))

    def select(self, rng: range) -> "MemmapDataset":
        """Select a subset of the dataset."""
        mmap = MemmapDataset.__new__(MemmapDataset)
        mmap.mmap = self.mmap[rng.start : rng.stop]
        return mmap

    def shard(self, num_shards: int, shard_id: int) -> "MemmapDataset":
        """Split the dataset into `num_shards` and return the `shard_id`-th shard."""
        mmap = MemmapDataset.__new__(MemmapDataset)

        # Split the mmap array into `num_shards` and return the `shard_id`-th shard
        shards = np.array_split(self.mmap, num_shards)
        mmap.mmap = shards[shard_id]
        return mmap


def compute_batches(lengths, max_tokens: int):
    """Split a list of lengths into batches that do not exceed `max_tokens`."""
    start = 0
    tokens_in_batch = 0
    batches = []

    for idx, length in enumerate(lengths):
        # Would adding this `length` exceed the capacity?
        if tokens_in_batch + length > max_tokens:
            # Close the previous batch: slice(start, idx)
            batches.append(slice(start, idx))

            # Start a new batch _with_ this item
            start = idx
            tokens_in_batch = length
        else:
            # It fits, so accumulate and keep going
            tokens_in_batch += length

    # Add the last batch if it has any items
    if start < len(lengths):
        batches.append(slice(start, len(lengths)))

    return batches


def load_index(root_dir: str) -> Dataset:
    """
    Walk `root_dir`, find all subdirectories matching 'rank_{integer}.idx',
    load the HF dataset from each, and concatenate them in ascending order
    of the integer.
    """
    pattern = re.compile(r"rank_(\d+)\.idx$")
    ranked_dirs = []

    # Traverse the directory tree
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            match = pattern.match(dirname)
            if match:
                rank = int(match.group(1))
                full_path = os.path.join(dirpath, dirname)
                ranked_dirs.append((rank, full_path))

    # Sort by the extracted integer
    ranked_dirs.sort(key=lambda x: x[0])

    # Load each dataset and collect into a list
    datasets_list = []
    for rank, ds_path in ranked_dirs:
        ds = load_from_disk(ds_path)
        datasets_list.append(ds)

    # Concatenate all datasets into one
    if not datasets_list:
        raise RuntimeError(
            f"No subdirectories matching 'rank_{{integer}}.idx' found under {root_dir}"
        )

    concatenated = concatenate_datasets(datasets_list)
    return concatenated


def pad_and_tensor(
    sequences: list[list[int]],
    labels: list[list[int]] | None = None,
    *,
    padding_value: int = 0,
    dtype: torch.dtype | None = torch.long,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of sequences to the same length and convert them to tensors.
    Returns a tuple of padded sequences and labels. The labels are the same as the
    sequences, but with -100 for the padding positions, which is useful for ignoring
    padding in loss calculations.
    """
    if labels is None:
        labels = sequences

    # find max length
    max_len = max(len(seq) for seq in sequences)
    # pad each sequence
    padded = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
    labels = [label + [-100] * (max_len - len(label)) for label in labels]

    # convert to tensor
    padded_tokens = torch.tensor(padded, dtype=dtype, device=device)
    padded_labels = torch.tensor(labels, dtype=dtype, device=device)
    return padded_tokens, padded_labels


def tokenize(batch: dict, *, args: IndexConfig, tokenizer):
    """Tokenize a batch of data with `tokenizer` according to `args`."""
    kwargs = dict(
        return_attention_mask=False,
        return_length=True,
    )
    if args.completion_column:
        # We're dealing with a prompt-completion dataset
        convos = [
            [
                {"role": "user", "content": assert_type(str, prompt)},
                {"role": "assistant", "content": assert_type(str, resp)},
            ]
            for prompt, resp in zip(
                batch[args.prompt_column], batch[args.completion_column]
            )
        ]
    elif args.conversation_column:
        # We're dealing with a conversation dataset
        convos = assert_type(list, batch[args.conversation_column])
    else:
        # We're dealing with vanilla next-token prediction
        return tokenizer(batch[args.prompt_column], truncation=True, **kwargs)

    # Make sure we only compute loss on the assistant's responses
    strings = tokenizer.apply_chat_template(convos, tokenize=False)
    encodings = tokenizer(strings, truncation=True, **kwargs)
    labels_list: list[list[int]] = []

    for i, convo in enumerate(convos):
        # Find the spans of the assistant's responses in the tokenized output
        pos = 0
        spans: list[tuple[int, int]] = []

        for msg in convo:
            if msg["role"] != "assistant":
                continue

            ans = msg["content"]
            start = strings[i].find(ans, pos)
            pos = start + len(ans)  # move past this match

            start_token = encodings.char_to_token(i, start)
            end_token = encodings.char_to_token(i, pos)
            spans.append((start_token, end_token))

        # Labels are -100 everywhere except where the assistant's response is
        tokens = encodings["input_ids"][i]
        labels = [-100] * len(tokens)
        for start, end in spans:
            if start is not None and end is not None:
                labels[start:end] = tokens[start:end]

        labels_list.append(labels)

    return dict(**encodings, labels=labels_list)


def unflatten(x: torch.Tensor, shapes: dict[str, Sequence[int]], dim: int = -1):
    """Unflatten a tensor `x` into a dictionary of tensors with specified shapes."""
    numels = [math.prod(shape) for shape in shapes.values()]
    return {
        name: x.unflatten(dim, shape)
        for (name, shape), x in zip(shapes.items(), x.split(numels, dim=dim))
    }
