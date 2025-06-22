import os
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel

from bergson.data import IndexConfig, compute_batches
from bergson.distributed import distributed_computing
from bergson.gradients import GradientProcessor
from bergson.hessians.covariance_all_factors import (
    compute_covariance,
    compute_eigendecomposition,
    compute_eigenvalue_correction,
)
from bergson.utils import get_layer_list


def worker_ekfac(rank: int, world_size: int, cfg: IndexConfig, ds: Dataset):
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

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    if cfg.fsdp:
        # Shard each individual transformer layer
        for layer in get_layer_list(model):
            fully_shard(layer)

        # Shard the entire model
        fully_shard(model)

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

    if os.path.exists(cfg.ekfac_path):
        if rank == 0:
            print(f"Loading matrices from '{cfg.ekfac_path}'")

        processor = GradientProcessor.load(
            cfg.processor_path,
            map_location=f"cuda:{rank}",
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
    compute_all_factors(
        model,
        ds,
        processor,
        cfg.run_path,
        batches=batches,
        target_modules=target_modules,
    )


def compute_all_factors(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[slice] | None = None,
    target_modules: set[str] | None = None,
):
    compute_covariance(
        model,
        data,
        processor,
        path,
        batches=batches,
        target_modules=target_modules,
    )

    compute_eigendecomposition(
        path,
    )

    compute_eigenvalue_correction(
        model,
        data,
        processor,
        path,
        batches=batches,
        target_modules=target_modules,
    )

    pass


def compute_EKFAC(cfg: IndexConfig):
    distributed_computing(cfg=cfg, worker_fn=worker_ekfac)
