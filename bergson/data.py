import json
import math
import os
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pyarrow as pa
import torch
import torch.distributed as dist
from datasets import Dataset
from numpy.typing import DTypeLike
from simple_parsing import field

from .utils import assert_type


@dataclass
class DataConfig:
    dataset: str = "EleutherAI/SmolLM2-135M-10B"
    """Dataset identifier to build the index from."""

    prompt_column: str = "text"
    """Column in the dataset that contains the prompts."""

    completion_column: str = ""
    """Optional column in the dataset that contains the completions."""

    conversation_column: str = ""
    """Optional column in the dataset that contains the conversation."""


@dataclass
class IndexConfig:
    """Config for building the index and running the model/dataset pipeline."""

    run_path: str = field(positional=True)
    """Name of the run. Used to create a directory for the index."""

    data: DataConfig = field(default_factory=DataConfig)
    """Specification of the data on which to build the index."""

    model: str = "HuggingFaceTB/SmolLM2-135M"
    """Name of the model to load."""

    fsdp: bool = False
    """Whether to use Fully Sharded Data Parallel (FSDP) for collecing gradients."""

    precision: Literal["bf16", "fp16", "fp32", "int4", "int8"] = "bf16"
    """Precision to use for the model parameters."""

    projection_dim: int = 16
    """Dimension of the random projection for the index, or 0 to disable it."""

    token_batch_size: int = 8192
    """Batch size in tokens for building the index."""

    normalizer: Literal["adafactor", "adam", "none"] = "adafactor"
    """Type of normalizer to use for the gradients."""

    fisher_fourth_root: bool = False
    """Whether to use the fourth root of the Fisher information matrix."""

    processor_path: str = ""
    """Path to a precomputed processor."""

    skip_preconditioners: bool = False
    """Whether to skip computing preconditioners for the gradients."""

    stats_sample_size: int = 1_000
    """Number of examples to use for estimating processor statistics."""

    drop_columns: bool = False
    """Only return the new dataset columns."""

    ekfac_path: str = ""
    """Path to the EKFAC computation, if it exists."""


def ceildiv(a: int, b: int) -> int:
    """Ceiling division of two integers."""
    return -(-a // b)  # Equivalent to math.ceil(a / b) but faster for integers


def compute_batches(lengths, max_tokens: int):
    """Split a list of lengths into batches that do not exceed `max_tokens`.

    Each batch is intended to be sharded into roughly equal parts across all available
    processes in a distributed setup. Accordingly, if `torch.distributed` is
    initialized, the batches are sliced to ensure that each process only works on its
    own subset of the data.
    """
    start = 0
    tokens_in_batch = 0
    batches = []

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    for idx, length in enumerate(lengths):
        # Would adding this `length` exceed the capacity?
        if tokens_in_batch + length > max_tokens:
            batch = slice(start, idx)
            if lengths[batch]:
                batches.append(batch)

            # Start a new batch with this item
            start = idx
            tokens_in_batch = length
        else:
            # It fits, so accumulate and keep going
            tokens_in_batch += length

    # Add the last batch if it has any items
    if start < len(lengths):
        batch = slice(start, len(lengths))
        if lengths[batch]:
            batches.append(batch)

    min_batches = ceildiv(len(batches), world_size)
    local_batches = batches[rank::world_size]
    if len(local_batches) < min_batches:
        # Split the last batch in two
        last_batch = local_batches[-1]
        batch_len = last_batch.stop - last_batch.start

        if batch_len <= 2:
            raise RuntimeError(
                "Unable to evenly split data into batches. Please pad the dataset to a multiple of the world size."
            )

        local_batches[-1] = slice(last_batch.start, last_batch.start + batch_len // 2)
        local_batches.append(slice(last_batch.start + batch_len // 2, last_batch.stop))

    assert len(local_batches) == min_batches
    return local_batches


def create_index(root: str, dtype: DTypeLike, shape: tuple[int, ...]) -> np.memmap:
    """Create a memory-mapped file for storing gradients, and persist metadata."""
    grad_path = os.path.join(root, "gradients.bin")
    rank = dist.get_rank() if dist.is_initialized() else 0

    # ── 1. Rank-0 creates file & metadata exactly once ─────────────────────────
    if rank == 0:
        # Ensure the directory exists
        os.makedirs(root, exist_ok=True)

        # Allocate (extends file to right size without writing zeros byte-by-byte)
        nbytes = np.dtype(dtype).itemsize * int(np.prod(shape))
        with open(grad_path, "wb") as f:
            f.truncate(nbytes)

            # Force the directory entry + data to disk *before* other ranks continue
            os.fsync(f.fileno())

        # Persist metadata for future runs
        with open(root + "/info.json", "w") as f:
            json.dump({"grad_size": shape[1], "num_grads": shape[0]}, f, indent=2)

    # 2. Everyone blocks until the file is definitely there & sized
    if dist.is_initialized():
        dist.barrier()

    return np.memmap(grad_path, dtype=dtype, mode="w+", shape=shape)


def load_gradients(root_dir: str) -> np.memmap:
    """Map the gradients stored in `root_dir` into memory."""
    with open(os.path.join(root_dir, "info.json")) as f:
        info = json.load(f)
        grad_size = info["grad_size"]
        num_grads = info["num_grads"]

    mmap = np.memmap(
        root_dir + "/gradients.bin",
        dtype=np.float16,
        mode="r",
        shape=(num_grads, grad_size),
    )
    return mmap


def load_gradient_dataset(root_dir: str) -> Dataset:
    """Load a dataset of gradients from `root_dir`."""
    mmap = load_gradients(root_dir)
    flat = pa.array(mmap.reshape(-1))
    col = pa.FixedSizeListArray.from_arrays(flat, mmap.shape[1])

    # Create a Dataset with the gradients as a single column
    ds = Dataset.load_from_disk(root_dir + "/data.hf")
    return ds.add_column("gradients", col, new_fingerprint="grads")


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


def tokenize(batch: dict, *, args: DataConfig, tokenizer):
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
            for prompt, resp in zip(batch[args.prompt_column], batch[args.completion_column])
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
    return {name: x.unflatten(dim, shape) for (name, shape), x in zip(shapes.items(), x.split(numels, dim=dim))}
