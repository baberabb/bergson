import socket
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset, load_dataset
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from transformers import AutoTokenizer

from .data import IndexConfig, tokenize
from .utils import assert_type


def worker_wrapper(rank: int, world_size: int, cfg: IndexConfig, ds: Dataset, worker_fn: Callable):
    try:
        worker_fn(rank, world_size, cfg, ds)
    finally:
        dist.destroy_process_group()


def distributed_computing(cfg: IndexConfig, worker_fn: Callable):
    mp.set_sharing_strategy("file_system")

    # Do all the data loading and preprocessing on the main process
    data_str = cfg.data.dataset
    if data_str.endswith(".csv"):
        ds = assert_type(Dataset, Dataset.from_csv(data_str))
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = assert_type(Dataset, Dataset.from_json(data_str))
    else:
        try:
            ds = assert_type(Dataset, load_dataset(data_str, split="train"))
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                ds = Dataset.load_from_disk(data_str, keep_in_memory=False)
            else:
                raise e

    metadata = {"length"}
    if cfg.drop_columns:
        metadata |= set(ds.column_names)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    ds = ds.map(lambda _, idx: dict(_row=idx), with_indices=True).shuffle(seed=42)
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer),
    )
    ds = ds.sort("length", reverse=True)

    # Find an available port for distributed training
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        _, port = s.getsockname()

    world_size = torch.cuda.device_count()

    ctx = start_processes(
        "build",
        worker_wrapper,
        args={i: (i, world_size, cfg, ds, worker_fn) for i in range(world_size)},
        envs={
            i: {
                "LOCAL_RANK": str(i),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(port),
            }
            for i in range(world_size)
        },
        logs_specs=DefaultLogsSpecs(),
    )
    ctx.wait()
