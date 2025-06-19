import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset, load_dataset
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data import IndexConfig, compute_batches, tokenize
from .gradients import GradientProcessor
from .processing import collect_gradients, fit_normalizers
from .utils import assert_type, get_layer_list, patch_fsdp_int8_model, post_patch_fsdp_int8_model


def _worker(rank: int, world_size: int, cfg: IndexConfig, ds: Dataset):
    # These should be set by the main process
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
    torch.cuda.set_device(rank)

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
    )

    if cfg.precision == "int8" and cfg.fsdp:
        patch_fsdp_int8_model(model, dtype)

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    if cfg.fsdp:
        # Shard each individual transformer layer
        for layer in get_layer_list(model):
            fully_shard(layer)

        # Shard the entire model
        fully_shard(model)

    if cfg.precision == "int8" and cfg.fsdp:
        post_patch_fsdp_int8_model(model)

    # Check for PEFT adapters
    try:
        adapters = model.active_adapters()
    except ValueError:
        target_modules = None
    else:
        if rank == 0:
            print("PEFT model detected.")

        target_modules = set()

        for adapter_name in adapters:
            state = model.get_adapter_state_dict(adapter_name)

            for name in state:
                prefix = name.removesuffix(".weight")
                name = prefix + "." + adapter_name

                try:
                    model.get_submodule(name)
                except AttributeError:
                    print(f"Adapter parameter '{name}' not found in the model.")

                target_modules.add(name.removeprefix("model."))

    batches = compute_batches(ds["length"], cfg.token_batch_size)

    try:
        if GradientProcessor.exists(cfg.run_path):
            if rank == 0:
                print(f"Loading processor from '{cfg.run_path}'")

            processor = GradientProcessor.load(
                cfg.run_path,
                map_location=f"cuda:{rank}",
            )
        else:
            if cfg.normalizer != "none":
                normalizers = fit_normalizers(
                    model,
                    ds,
                    batches=batches,
                    kind=cfg.normalizer,
                    max_documents=cfg.stats_sample_size or None,
                    target_modules=target_modules,
                )
            else:
                normalizers = {}

        processor = GradientProcessor(
            normalizers,
            fisher_fourth_root=cfg.fisher_fourth_root,
            projection_dim=cfg.projection_dim or None,
        )
        if rank == 0:
            processor.save(cfg.run_path)

    batches = compute_batches(ds["length"], cfg.token_batch_size)
    collect_gradients(
        model,
        ds,
        processor,
        cfg.run_path,
        batches=batches,
        skip_preconditioners=cfg.skip_preconditioners,
        target_modules=target_modules,
    )


def worker(rank: int, world_size: int, cfg: IndexConfig, ds: Dataset):
    try:
        _worker(rank, world_size, cfg, ds)
    finally:
        dist.destroy_process_group()


def build_index(cfg: IndexConfig):
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
    # ds = ds.select(list(range(73)))
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
        worker,
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
