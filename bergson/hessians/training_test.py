import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from simple_parsing import parse
from torch.distributed.fsdp import fully_shard
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from bergson.data import IndexConfig, allocate_batches, pad_and_tensor, tokenize
from bergson.gradients import (
    Normalizer,
)
from bergson.utils import assert_type, get_layer_list

NORMALIZER_TYPES: dict[str, type["Normalizer"]] = {}

NORMALIZER_TYPES: dict[str, type["Normalizer"]] = {}


def setup_distributed():
    """Properly initialize distributed training"""
    try:
        # Check if we're in a distributed environment
        if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Initialize the process group first
            dist.init_process_group(backend="nccl")

            # Set CUDA device for current rank
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            print("Distributed training initialized successfully")
            return dist.get_rank(), dist.get_world_size()
        else:
            # Not in distributed environment
            raise RuntimeError("Not in distributed environment")
    except:
        # Fall back to single GPU
        print("Falling back to single GPU")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1


def train(
    model: PreTrainedModel,
    data: Dataset,
    *,
    batches: list[list[int]] | None = None,
):
    rank = dist.get_rank() if dist.is_initialized() else 0
    for sl in tqdm(batches, disable=rank != 0, desc="Computing covariances"):
        batch = data[sl]
        x, y = pad_and_tensor(
            batch["input_ids"],  # type: ignore
            labels=batch.get("labels"),  # type: ignore
            device=model.device,
        )

        logits = model(x).logits
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            y[:, 1:].flatten(),
            reduction="none",
        ).reshape_as(y[:, 1:])

        masks = y[:, 1:] != -100
        denoms = masks.sum(dim=1, dtype=logits.dtype)
        losses = losses.sum(1).div(denoms)
        print(losses.mean().item())
        losses.mean().backward()

        model.zero_grad()

    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # For multi-GPU
    np.random.seed(42)
    random.seed(42)
    # 1. Initialize distributed FIRST
    rank, world_size = setup_distributed()

    # 2. Initialize your checkpoint manager and task
    cfg = parse(IndexConfig)

    match cfg.precision:
        case "bf16":
            dtype = torch.bfloat16
        case "fp16":
            dtype = torch.float16
        case "fp32":
            dtype = torch.float32
        case "int4" | "int8":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        case other:
            raise ValueError(f"Unsupported precision: {other}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map={"": f"cuda:{rank}" if not cfg.fsdp else "cpu"},
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=cfg.precision == "int4",
                load_in_8bit=cfg.precision == "int8",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_storage=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if cfg.precision in ("int4", "int8")
            else None
        ),
        torch_dtype=dtype,
        revision=cfg.revision,
    )

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    if cfg.fsdp:
        # Shard each individual transformer layer
        for layer in get_layer_list(model):
            fully_shard(layer)

        # Shard the entire model
        fully_shard(model)

    data_str = cfg.data.dataset
    if data_str.endswith(".csv"):
        ds = assert_type(Dataset, Dataset.from_csv(data_str))
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = assert_type(Dataset, Dataset.from_json(data_str))
    else:
        try:
            ds = load_dataset(data_str, split="train", streaming=cfg.streaming)

            if isinstance(ds, DatasetDict) or isinstance(ds, IterableDatasetDict):
                raise NotImplementedError("DatasetDicts and IterableDatasetDicts are not supported.")
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                ds = Dataset.load_from_disk(data_str, keep_in_memory=False)
            else:
                raise e

    remove_columns = ds.column_names if cfg.drop_columns else None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, model_max_length=cfg.token_batch_size, revision=cfg.revision)
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer),
        remove_columns=remove_columns,
    )
    assert isinstance(ds, Dataset), "Expected a single Dataset, not a DatasetDict or IterableDatasetDict"

    ds = ds.select(list(range(96)))
    batches = allocate_batches(ds["length"], cfg.token_batch_size)

    train(
        model,
        ds,
        batches=batches,
    )

    if dist.is_initialized():
        dist.destroy_process_group()
