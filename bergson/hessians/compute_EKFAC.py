import json
import os
from dataclasses import asdict
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import fully_shard
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel

from bergson.data import IndexConfig, allocate_batches
from bergson.distributed import distributed_computing
from bergson.gradients import GradientProcessor
from bergson.hessians.covariance_all_factors import EkfacComputer, compute_eigenvalue_correction
from bergson.utils import assert_type, get_layer_list


def worker_ekfac(rank: int, world_size: int, cfg: IndexConfig, ds: Dataset | IterableDataset):
    torch.cuda.set_device(rank)

    # These should be set by the main process
    if world_size >= 1:
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

    normalizers = {}

    processor = GradientProcessor(
        normalizers=normalizers,
        fisher_fourth_root=cfg.fisher_fourth_root,
        projection_dim=None,
    )
    if rank == 0:
        processor.save(cfg.run_path)

    if rank == 0:
        json.dump(asdict(cfg), open(os.path.join(cfg.run_path, "config.json"), "w"), indent=2)
    cfg.ekfac_path = os.path.join(cfg.run_path, "influence_results")
    os.makedirs(cfg.ekfac_path, exist_ok=True)

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"], cfg.token_batch_size)

        compute_all_factors(
            model,
            ds,
            processor,
            cfg.ekfac_path,
            batches=batches,
            target_modules=target_modules,
            debug=cfg.debug,
            profile=cfg.profile,
        )

    else:
        # Convert each chunk of the IterableDataset to Dataset then collect their gradients
        buf, chunk_id = [], 0

        def flush():
            nonlocal buf, chunk_id
            if not buf:
                return
            sub_ds = assert_type(Dataset, Dataset.from_list(buf))
            batches = allocate_batches(sub_ds["length"], cfg.token_batch_size)

            compute_all_factors(
                model,
                sub_ds,
                processor,
                os.path.join(cfg.run_path, f"chunk-{chunk_id:05d}"),
                batches=batches,
                target_modules=target_modules,
                debug=cfg.debug,
                profile=cfg.profile,
            )

            buf.clear()
            chunk_id += 1

        for ex in tqdm(ds, desc="Collecting gradients"):
            buf.append(ex)
            if len(buf) == cfg.streaming_chunk_size:
                flush()
        flush()


def compute_all_factors(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[list[int]],
    target_modules: set[str] | None = None,
    debug: bool = False,
    profile: bool = False,
):
    computer = EkfacComputer(
        model=model,
        processor=processor,
        data=data,
        path=path,
        batches=batches,
        target_modules=target_modules,
        debug=debug,
        profile=profile,
    )
    # computer.compute_covariance()

    # dist.barrier() if dist.is_initialized() else None

    # computer.compute_eigendecomposition(covariance_type="activation")
    # computer.compute_eigendecomposition(covariance_type="gradient")

    # dist.barrier() if dist.is_initialized() else None

    compute_eigenvalue_correction(
        model,
        data,
        processor,
        path,
        batches=batches,
        target_modules=target_modules,
    )


def compute_EKFAC(cfg: IndexConfig):
    distributed_computing(cfg=cfg, worker_fn=worker_ekfac)
