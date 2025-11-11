import json
import os
import shutil
import socket
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Callable

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
from peft import PeftConfig, PeftModel, get_peft_model_state_dict
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from torch.distributed.fsdp import fully_shard
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from bergson.data import DataConfig, IndexConfig, allocate_batches, tokenize
from bergson.gradients import GradientProcessor
from bergson.utils import assert_type, get_layer_list


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
            ds = load_dataset(data_str, split=cfg.data.split, streaming=cfg.streaming)

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


def setup_model_and_peft(
    cfg: IndexConfig,
    rank: int,
) -> tuple[AutoModelForCausalLM, set | None]:
    """Handle model loading, quantization, FSDP, and PEFT detection"""

    match cfg.precision:
        case "bf16":
            dtype = torch.bfloat16
        case "fp16":
            dtype = torch.float16
        case "fp32":
            dtype = torch.float32
        case "int4" | "int8":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        case "auto":
            dtype = "auto"
        case other:
            raise ValueError(f"Unsupported precision: {other}")

    # Common configuration
    device_map = {"": f"cuda:{rank}"} if not cfg.fsdp else "cpu"
    quantization_config = None
    if cfg.precision in ("int4", "int8"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=cfg.precision == "int4",
            load_in_8bit=cfg.precision == "int8",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Try to detect PEFT model
    try:
        peft_config = PeftConfig.from_pretrained(cfg.model)
    except ValueError:
        peft_config = None

    if peft_config is None:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
        )
        target_modules = None

    else:
        # Load PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # type: ignore
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
        )

        model = PeftModel.from_pretrained(
            base_model,
            cfg.model,
            device_map=device_map,
            autocast_adapter_dtype=False,
        )

        # Extract target modules
        target_modules = set()
        peft_state_dict = get_peft_model_state_dict(model=model)
        for adapter in model.peft_config.keys():
            for name in list(peft_state_dict.keys()):
                prefix = name.removesuffix(".weight")
                processed_name = f"{prefix}.{adapter}".removeprefix("base_model.")
                try:
                    model.get_submodule(processed_name)
                    target_modules.add(processed_name)
                except AttributeError:
                    print(
                        f"Adapter parameter '{processed_name}' not found in the model."
                    )

    # Configure gradients
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)  # type: ignore

    # Apply FSDP if needed
    if cfg.fsdp:
        for layer in get_layer_list(model):  # type: ignore
            fully_shard(layer)
        fully_shard(model)

    return model, target_modules  # type: ignore


def create_processor(
    cfg: IndexConfig,
    rank: int,
) -> GradientProcessor:
    """Handle processor creation and normalizer fitting"""
    if os.path.exists(cfg.processor_path):
        if rank == 0:
            print(f"Loading processor from '{cfg.processor_path}'")

        processor = GradientProcessor.load(
            Path(cfg.processor_path),
            map_location=f"cuda:{rank}",
        )
    else:
        processor = GradientProcessor(
            {},
            projection_dim=cfg.projection_dim or None,
            reshape_to_square=cfg.reshape_to_square,
            projection_type=cfg.projection_type,
            include_bias=cfg.include_bias,
        )
        if rank == 0:
            processor.save(cfg.partial_run_path)

    return processor


def worker_wrapper(
    rank: int,
    world_size: int,
    cfg: IndexConfig,
    ds: Dataset | IterableDataset,
    worker_fn: Callable,
    setup_model: bool = True,
    setup_processor: bool = True,
):
    try:
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

        # Initialize defaults for optional components
        model, target_modules, processor = None, None, None

        if setup_model:
            model, target_modules = setup_model_and_peft(cfg, rank)

        if setup_processor:
            if model is None:
                raise ValueError(
                    "Cannot create processor without model. "
                    "Set setup_model=True or provide model externally."
                )
            processor = create_processor(cfg, rank)

        if cfg.split_attention_modules:
            attention_cfgs = {
                module: cfg.attention for module in cfg.split_attention_modules
            }
        else:
            attention_cfgs = {}

        kwargs = {
            "model": model,
            "data": ds,
            "processor": processor,
            "cfg": cfg,
            "target_modules": target_modules,
            "attention_cfgs": attention_cfgs,
        }

        if setup_model and setup_processor:
            assert processor is not None
            if isinstance(ds, Dataset):
                batches = allocate_batches(ds["length"], cfg.token_batch_size)
                kwargs["batches"] = batches
                worker_fn(**kwargs)
            else:
                # Convert each shard to a Dataset then map over its gradients
                buf, shard_id = [], 0

                def flush(worker_fn, kwargs):
                    nonlocal buf, shard_id
                    if not buf:
                        return
                    ds_shard = assert_type(Dataset, Dataset.from_list(buf))
                    batches = allocate_batches(
                        ds_shard["length"][:], cfg.token_batch_size
                    )
                    kwargs["ds"] = ds_shard
                    kwargs["batches"] = batches
                    worker_fn(**kwargs)

                    buf.clear()
                    shard_id += 1

                for ex in tqdm(ds, desc="Collecting gradients"):
                    buf.append(ex)
                    if len(buf) == cfg.stream_shard_size:
                        flush(worker_fn=worker_fn, kwargs=kwargs)

                flush(worker_fn=worker_fn, kwargs=kwargs)  # Final flush
                if rank == 0:
                    processor.save(cfg.partial_run_path)
        else:
            worker_fn(**kwargs)

    finally:
        if dist.is_initialized():
            try:
                # Add a barrier to ensure all processes reach this point
                dist.barrier()
            except Exception:
                pass  # Ignore barrier failures during cleanup

            try:
                dist.destroy_process_group()
            except Exception:
                pass  # Ignore cleanup failures


def distributed_computing(
    cfg: IndexConfig,
    worker_fn: Callable,
    setup_data: bool = True,
    setup_model: bool = True,
    setup_processor: bool = True,
):
    os.makedirs(cfg.partial_run_path, exist_ok=True)
    with open(os.path.join(cfg.partial_run_path, "index_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=4)

    # Do all the data loading and preprocessing on the main process
    if setup_data:
        ds = setup_data_pipeline(cfg)
    else:
        # Create empty dataset for compatibility
        ds = assert_type(Dataset, Dataset.from_list([]))

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker_wrapper(0, 1, cfg, ds, worker_fn, setup_model, setup_processor)
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
                args={
                    i: (i, world_size, cfg, ds, worker_fn, setup_model, setup_processor)
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

    try:
        shutil.move(cfg.partial_run_path, cfg.run_path)
    except Exception:
        pass
