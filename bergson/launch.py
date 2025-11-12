import json
import os
import shutil
import socket
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
)

from bergson.collection import collect_gradients
from bergson.data import (
    DataConfig,
    IndexConfig,
    QueryConfig,
    allocate_batches,
    tokenize,
)
from bergson.query import get_query_grads
from bergson.score_writer import MemmapScoreWriter
from bergson.scorer import get_scorer
from bergson.utils import assert_type
from bergson.worker_utils import create_processor, setup_model_and_peft


def estimate_advantage(ds: Dataset, cfg: DataConfig):
    """Group rollouts by prompt and estimate advantages."""
    df = ds.select_columns([cfg.prompt_column, cfg.reward_column]).to_pandas()
    df = assert_type(pd.DataFrame, df)

    advantages = df[cfg.reward_column] - df.groupby(cfg.prompt_column)[
        cfg.reward_column
    ].transform("mean")

    return advantages.tolist()


def setup_data_pipeline(cfg: IndexConfig) -> Dataset | IterableDataset:
    """Handle data loading and preprocessing"""

    data_str = cfg.data.dataset
    if data_str.endswith(".csv"):
        ds = assert_type(Dataset, Dataset.from_csv(data_str))
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = assert_type(Dataset, Dataset.from_json(data_str))
    else:
        try:
            ds = load_dataset(
                data_str, split=cfg.data.split, streaming=cfg.data.streaming
            )

            if isinstance(ds, DatasetDict) or isinstance(ds, IterableDatasetDict):
                raise NotImplementedError(
                    "DatasetDicts and IterableDatasetDicts are not supported."
                )
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                ds = Dataset.load_from_disk(data_str, keep_in_memory=False)
            else:
                raise e

    # In many cases the token_batch_size may be smaller than the max length allowed by
    # the model. If cfg.data.truncation is True, we use the tokenizer to truncate
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    tokenizer.model_max_length = min(tokenizer.model_max_length, cfg.token_batch_size)

    remove_columns = ds.column_names if cfg.drop_columns else None

    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer),
        remove_columns=remove_columns,
    )

    if cfg.data.reward_column:
        assert isinstance(ds, Dataset), "Dataset required for advantage estimation"
        ds = ds.add_column(
            "advantage",
            estimate_advantage(ds, cfg.data),
            new_fingerprint="advantage",  # type: ignore
        )

    return ds


def worker(
    rank: int,
    world_size: int,
    cfg: IndexConfig,
    ds: Dataset | IterableDataset,
    query_cfg: QueryConfig | None = None,
):
    torch.cuda.set_device(rank)

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    model, target_modules = setup_model_and_peft(cfg, rank)
    processor = create_processor(cfg, rank)

    attention_cfgs = {module: cfg.attention for module in cfg.split_attention_modules}

    if query_cfg is not None:
        query_grads = get_query_grads(
            query_cfg, torch.device(f"cuda:{rank}"), torch.float32
        )
        num_scores = len(query_grads[query_cfg.modules[0]])

    kwargs = {
        "model": model,
        "data": ds,
        "processor": processor,
        "cfg": cfg,
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs,
    }

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"], cfg.token_batch_size)
        kwargs["batches"] = batches

        if query_cfg is not None:
            score_writer = MemmapScoreWriter(
                Path(query_cfg.scores_path),
                len(ds),
                num_scores,
                rank=rank,
            )
            scorer = get_scorer(
                query_grads,
                query_cfg,
                score_writer,
                cfg.module_wise,
                torch.device(f"cuda:{rank}"),
                torch.float32,
            )
            kwargs["scorer"] = scorer

        collect_gradients(**kwargs)
    else:
        # Convert each shard to a Dataset then map over its gradients
        buf, shard_id = [], 0

        def flush(kwargs):
            nonlocal buf, shard_id
            if not buf:
                return
            ds_shard = assert_type(Dataset, Dataset.from_list(buf))
            batches = allocate_batches(ds_shard["length"][:], cfg.token_batch_size)
            kwargs["ds"] = ds_shard
            kwargs["batches"] = batches

            if query_cfg is not None:
                score_writer = MemmapScoreWriter(
                    Path(query_cfg.scores_path) / f"shard-{shard_id:05d}",
                    len(ds_shard),
                    num_scores,
                    rank=rank,
                )
                scorer = get_scorer(
                    query_grads,
                    query_cfg,
                    score_writer,
                    cfg.module_wise,
                    torch.device(f"cuda:{rank}"),
                    torch.float32,
                )
                kwargs["scorer"] = scorer

            collect_gradients(**kwargs)

            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Collecting gradients"):
            buf.append(ex)
            if len(buf) == cfg.stream_shard_size:
                flush(kwargs=kwargs)

        flush(kwargs=kwargs)  # Final flush
        if rank == 0:
            processor.save(cfg.partial_run_path)


def dist_worker(
    worker: Callable,
    *worker_args,
):
    try:
        worker(*worker_args)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                print(f"Barrier failed during cleanup: {e}")
                pass

            dist.destroy_process_group()


def launch_distributed_run(process_name: str, worker, const_worker_args: list[Any]):
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, *const_worker_args)
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
                process_name,
                dist_worker,
                args={
                    i: (worker, i, world_size, *const_worker_args)
                    for i in range(world_size)
                },
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


def build(cfg: IndexConfig):
    cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    with (cfg.partial_run_path / "index_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    ds = setup_data_pipeline(cfg)

    launch_distributed_run("build", worker, [ds, cfg])

    shutil.move(cfg.partial_run_path, cfg.run_path)


def query(cfg: IndexConfig, query_cfg: QueryConfig):
    cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    with (cfg.partial_run_path / "index_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    ds = setup_data_pipeline(cfg)

    launch_distributed_run("query", worker, [ds, cfg, query_cfg])

    shutil.move(cfg.partial_run_path, cfg.run_path)
