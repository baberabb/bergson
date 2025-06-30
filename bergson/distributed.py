import socket
import sys
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
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
    # Do all the data loading and preprocessing on the main process
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

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker_fn(0, 1, cfg, ds)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = None
        try:
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
        finally:
            if ctx is not None:
                ctx.close()  # Kill any processes that are still running
                sys.exit(0)
