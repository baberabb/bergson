import os
import socket
from datetime import timedelta
from typing import cast

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset, IterableDataset
from peft import PeftConfig, PeftModel
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from torch.distributed.fsdp import fully_shard
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from .collection import collect_gradients
from .data import DataConfig, IndexConfig, allocate_batches, load_data_string, tokenize
from .gradients import GradientProcessor
from .peft import detect_peft_modules
from .utils import assert_type, get_layer_list


def worker(
    rank: int,
    world_size: int,
    cfg: IndexConfig,
    ds: Dataset | IterableDataset,
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
            revision=cfg.revision,
        )
        target_modules = None

    else:
        # Load PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # type: ignore
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
            revision=cfg.revision,
        )

        model = PeftModel.from_pretrained(
            base_model,
            cfg.model,
            device_map=device_map,
            autocast_adapter_dtype=False,
        )
        target_modules = detect_peft_modules(model)

        # Hack for type checking
        model = cast(PreTrainedModel, model)

    if rank == 0:
        print(f"Model loaded with dtype: {model.dtype}")

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    if cfg.fsdp:
        # Shard each individual transformer layer
        for layer in get_layer_list(model):
            fully_shard(layer)

        # Shard the entire model
        fully_shard(model)

    if os.path.exists(cfg.processor_path):
        if rank == 0:
            print(f"Loading processor from '{cfg.processor_path}'")

        processor = GradientProcessor.load(
            cfg.processor_path,
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
        if rank == 0 and cfg.save_processor:
            processor.save(cfg.partial_run_path)

    if cfg.split_attention_modules:
        attention_cfgs = {
            module: cfg.attention for module in cfg.split_attention_modules
        }
    else:
        attention_cfgs = {}

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"][:], cfg.token_batch_size)
        collect_gradients(
            model,
            ds,
            processor,
            cfg.partial_run_path,
            batches=batches,
            kl_divergence=cfg.loss_fn == "kl",
            loss_reduction=cfg.loss_reduction,
            skip_preconditioners=cfg.skip_preconditioners,
            target_modules=target_modules,
            attention_cfgs=attention_cfgs,
            save_index=cfg.save_index,
            save_processor=cfg.save_processor,
            drop_columns=cfg.drop_columns,
        )
    else:
        # Convert each shard to a Dataset then map over its gradients
        buf, shard_id = [], 0

        def flush():
            nonlocal buf, shard_id
            if not buf:
                return
            ds_shard = assert_type(Dataset, Dataset.from_list(buf))
            batches = allocate_batches(ds_shard["length"][:], cfg.token_batch_size)
            collect_gradients(
                model,
                ds_shard,
                processor,
                os.path.join(cfg.partial_run_path, f"shard-{shard_id:05d}"),
                batches=batches,
                kl_divergence=cfg.loss_fn == "kl",
                loss_reduction=cfg.loss_reduction,
                skip_preconditioners=cfg.skip_preconditioners,
                target_modules=target_modules,
                attention_cfgs=attention_cfgs,
                save_index=cfg.save_index,
                # Save a processor state checkpoint after each shard
                save_processor=cfg.save_processor,
                drop_columns=cfg.drop_columns,
            )
            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Collecting gradients"):
            buf.append(ex)
            if len(buf) == cfg.stream_shard_size:
                flush()
        flush()

        if cfg.save_processor:
            processor.save(cfg.partial_run_path)


def dist_worker(rank: int, world_size: int, cfg: IndexConfig, ds: Dataset):
    try:
        worker(rank, world_size, cfg, ds)
    finally:
        dist.destroy_process_group()


def estimate_advantage(ds: Dataset, cfg: DataConfig):
    """Group rollouts by prompt and estimate advantages."""
    df = ds.select_columns([cfg.prompt_column, cfg.reward_column]).to_pandas()
    df = assert_type(pd.DataFrame, df)

    advantages = df[cfg.reward_column] - df.groupby(cfg.prompt_column)[
        cfg.reward_column
    ].transform("mean")

    return advantages.tolist()


def build_gradient_dataset(cfg: IndexConfig):
    # In many cases the token_batch_size may be smaller than the max length allowed by
    # the model. If cfg.data.truncation is True, we use the tokenizer to truncate
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    tokenizer.model_max_length = min(tokenizer.model_max_length, cfg.token_batch_size)

    # Do all the data loading and preprocessing on the main process
    ds = load_data_string(cfg.data.dataset, cfg.data.split, streaming=cfg.streaming)

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

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, cfg, ds)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = start_processes(
            "build",
            dist_worker,
            args={i: (i, world_size, cfg, ds) for i in range(world_size)},
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

    try:
        os.rename(cfg.partial_run_path, cfg.run_path)
    except Exception:
        pass
