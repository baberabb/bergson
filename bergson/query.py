import json
import os
import socket
from copy import deepcopy
from datetime import timedelta
from typing import cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset, IterableDataset
from peft import PeftConfig, PeftModel
from torch import Tensor
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from torch.distributed.fsdp import fully_shard
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from .build import build_gradient_dataset, estimate_advantage
from .data import (
    IndexConfig,
    QueryConfig,
    allocate_batches,
    load_data_string,
    load_gradient_dataset,
    load_gradients,
    tokenize,
)
from .gradients import GradientProcessor
from .peft import detect_peft_modules
from .scan import scan_gradients
from .utils import assert_type, get_layer_list


def get_query_data(index_cfg: IndexConfig, query_cfg: QueryConfig):
    """
    Assemble the preconditioners for the query dataset. Because they are tensors,
    they can be combined according to https://arxiv.org/html/2410.17413v1
    """
    # Save the query gradients if they don't already exist
    if not os.path.exists(query_cfg.run_path):
        # Create a copy of the index configuration that uses the query dataset
        cfg = deepcopy(index_cfg)
        cfg.data = deepcopy(query_cfg.query_data)
        cfg.run_path = query_cfg.run_path

        if query_cfg.save_processor or query_cfg.apply_query_preconditioner != "none":
            cfg.save_processor = True
        cfg.save_index = True

        build_gradient_dataset(cfg)

    # Apply specified preconditioners to the query dataset
    mixed_preconditioner = {}
    if query_cfg.apply_query_preconditioner != "none":
        if query_cfg.query_preconditioner_path is not None:
            query_processor = GradientProcessor.load(
                query_cfg.query_preconditioner_path, map_location="cuda"
            )
        else:
            query_processor = GradientProcessor.load(
                query_cfg.run_path, map_location="cuda"
            )

        for name, preconditioner in query_processor.preconditioners.items():
            mixed_preconditioner[name] = preconditioner
            if query_cfg.apply_index_preconditioner != "none":
                mixed_preconditioner[name] *= query_cfg.mixing_coefficient

    if query_cfg.apply_index_preconditioner != "none":
        if query_cfg.index_preconditioner_path is not None:
            index_processor = GradientProcessor.load(
                query_cfg.index_preconditioner_path, map_location="cuda"
            )
        else:
            print(
                "Building index dataset gradient processor. "
                "Warning: this will take as long as the query scan itself."
            )
            build_gradient_dataset(index_cfg)
            processor_dir = index_cfg.processor_path or index_cfg.run_path
            index_processor = GradientProcessor.load(processor_dir, map_location="cuda")

        for name, preconditioner in index_processor.preconditioners.items():
            if query_cfg.apply_query_preconditioner != "none":
                preconditioner *= 1 - query_cfg.mixing_coefficient
                mixed_preconditioner[name] += preconditioner
            else:
                mixed_preconditioner[name] = preconditioner

    with open(os.path.join(query_cfg.run_path, "info.json"), "r") as f:
        target_modules = json.load(f)["dtype"]["names"]

    # Load the query dataset
    query_ds = load_gradient_dataset(query_cfg.run_path, concatenate_gradients=False)
    query_ds = query_ds.with_format("torch", columns=target_modules)

    if mixed_preconditioner:

        def precondition(batch):
            for name in target_modules:
                preconditioner = mixed_preconditioner[name].to(
                    device=batch.device, dtype=batch.dtype
                )
                batch[name] = (batch[name].cuda() @ preconditioner).cpu()

            return batch

        query_ds = query_ds.map(
            precondition, batched=True, batch_size=query_cfg.batch_size
        )

    return query_ds


def build_query_callback(
    query_cfg: QueryConfig,
    query_ds: Dataset,
    *,
    grad_dtype: torch.dtype,
    grad_device: torch.device | str,
):
    # TODO do we want to support returning per-module scores?

    """
    Build a function to query the gradients on-the-fly.
    """

    target_device = torch.device(grad_device)

    if not query_cfg.modules:
        query_cfg.modules = load_gradients(query_cfg.run_path).dtype.names

    query_ds = query_ds.with_format("torch", columns=query_cfg.modules)

    def get_mean_query_callback():
        acc = {
            module: torch.zeros_like(
                query_ds[0][module], device=target_device, dtype=torch.float32
            )
            for module in query_cfg.modules
        }

        def sum_(*cols):
            for module, x in zip(query_cfg.modules, cols):
                if query_cfg.unit_normalize:
                    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)  # avoid div-by-zero
                acc[module] += x.to(device=target_device, dtype=torch.float32).sum(0)

        query_ds.map(
            sum_,
            input_columns=query_cfg.modules,
            batched=True,
            batch_size=query_cfg.batch_size,
        )

        callback_query = {
            module: (acc[module] / len(query_ds)).to(
                device=target_device, dtype=grad_dtype
            )
            for module in query_cfg.modules
        }

        @torch.inference_mode()
        def query_callback(mod_grads: dict[str, torch.Tensor]):
            nonlocal callback_query
            # Cat grads across modules
            # grads = torch.cat([mod_grads[name] for name in query_cfg.modules], dim=1)
            scores = torch.stack(
                [
                    mod_grads[module] @ callback_query[module]
                    for module in query_cfg.modules
                ],
                dim=-1,
            ).sum(-1)
            return scores

        return query_callback

    def get_nearest_query_callback():
        queries = assert_type(Tensor, query_ds["gradients"]).to(
            device=target_device, dtype=grad_dtype
        )

        if query_cfg.unit_normalize:
            queries /= queries.norm(dim=1, keepdim=True)

        def query_callback(mod_grads: dict[str, torch.Tensor]):
            grads = torch.cat([mod_grads[name] for name in query_cfg.modules], dim=0)
            if query_cfg.unit_normalize:
                grads /= grads.norm(dim=1, keepdim=True)

            # Calculate scores as the max of the inner products with the queries
            all_scores = grads @ queries.T
            return all_scores.max(dim=-1).values

        return query_callback

    if query_cfg.query_method == "mean":
        return get_mean_query_callback()
    elif query_cfg.query_method == "nearest":
        return get_nearest_query_callback()
    else:
        raise ValueError(f"Invalid query method: {query_cfg.query_method}")


def worker(
    rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    query_cfg: QueryConfig,
    ds: Dataset | IterableDataset,
    query_ds: Dataset,
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

    match index_cfg.precision:
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

    device_map = {"": f"cuda:{rank}"} if not index_cfg.fsdp else "cpu"
    quantization_config = None
    if index_cfg.precision in ("int4", "int8"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=index_cfg.precision == "int4",
            load_in_8bit=index_cfg.precision == "int8",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Try to detect PEFT model
    try:
        peft_config = PeftConfig.from_pretrained(index_cfg.model)
    except ValueError:
        peft_config = None

    if peft_config is None:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            index_cfg.model,
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
            revision=index_cfg.revision,
        )
        target_modules = None

    else:
        # Load PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # type: ignore
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
            revision=index_cfg.revision,
        )

        model = PeftModel.from_pretrained(
            base_model,
            index_cfg.model,
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

    if index_cfg.fsdp:
        # Shard each individual transformer layer
        for layer in get_layer_list(model):
            fully_shard(layer)

        # Shard the entire model
        fully_shard(model)

    processor_dir = index_cfg.processor_path or index_cfg.run_path
    processor_cfg_path = os.path.join(processor_dir, "processor_config.json")

    if os.path.exists(processor_cfg_path):
        if rank == 0:
            print(f"Loading processor from '{processor_dir}'")

        processor = GradientProcessor.load(
            processor_dir,
            map_location=f"cuda:{rank}",
        )
    else:
        processor = GradientProcessor(
            {},
            projection_dim=index_cfg.projection_dim or None,
            reshape_to_square=index_cfg.reshape_to_square,
            projection_type=index_cfg.projection_type,
        )
        if rank == 0 and index_cfg.save_processor:
            processor.save(index_cfg.partial_run_path)

    query_callback = build_query_callback(
        query_cfg,
        query_ds,
        grad_dtype=dtype if dtype != "auto" else torch.float16,
        grad_device=f"cuda:{rank}",
    )
    print("Scanning gradients with save_index:", index_cfg.save_index)

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"][:], index_cfg.token_batch_size)
        scan_gradients(
            model,
            ds,
            processor,
            index_cfg.partial_run_path,
            batches=batches,
            kl_divergence=index_cfg.loss_fn == "kl",
            loss_reduction=index_cfg.loss_reduction,
            skip_preconditioners=index_cfg.skip_preconditioners,
            target_modules=target_modules,
            head_cfgs=index_cfg.head_cfgs,
            drop_columns=index_cfg.drop_columns,
            query_callback=query_callback,
            save_index=index_cfg.save_index,
            save_processor=index_cfg.save_processor,
        )
    else:
        # Convert each shard to a Dataset then collect its gradients
        buf, shard_id = [], 0

        def flush():
            nonlocal buf, shard_id
            if not buf:
                return
            ds_shard = assert_type(Dataset, Dataset.from_list(buf))
            batches = allocate_batches(
                ds_shard["length"][:], index_cfg.token_batch_size
            )
            scan_gradients(
                model,
                ds_shard,
                processor,
                os.path.join(index_cfg.partial_run_path, f"shard-{shard_id:05d}"),
                batches=batches,
                kl_divergence=index_cfg.loss_fn == "kl",
                loss_reduction=index_cfg.loss_reduction,
                skip_preconditioners=index_cfg.skip_preconditioners,
                target_modules=target_modules,
                head_cfgs=index_cfg.head_cfgs,
                drop_columns=index_cfg.drop_columns,
                query_callback=query_callback,
                save_index=index_cfg.save_index,
                save_processor=index_cfg.save_processor,
            )
            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Querying gradients on the fly"):
            buf.append(ex)
            if len(buf) == index_cfg.stream_shard_size:
                flush()
        flush()


def dist_worker(
    rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    query_cfg: QueryConfig,
    ds: Dataset,
    query_ds: Dataset,
):
    try:
        worker(rank, world_size, index_cfg, query_cfg, ds, query_ds)
    finally:
        dist.destroy_process_group()


def query_gradient_dataset(query_cfg: QueryConfig, index_cfg: IndexConfig):
    # In many cases the token_batch_size may be smaller than the max length allowed by
    # the model. If cfg.data.truncation is True, we use the tokenizer to truncate
    tokenizer = AutoTokenizer.from_pretrained(
        index_cfg.model, revision=index_cfg.revision
    )
    tokenizer.model_max_length = min(
        tokenizer.model_max_length, index_cfg.token_batch_size
    )

    # Do all the data loading and preprocessing on the main process
    ds = load_data_string(
        index_cfg.data.dataset, index_cfg.data.split, streaming=index_cfg.streaming
    )

    remove_columns = ds.column_names if index_cfg.drop_columns else None
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=index_cfg.data, tokenizer=tokenizer),
        remove_columns=remove_columns,
    )
    if index_cfg.data.reward_column:
        assert isinstance(ds, Dataset), "Dataset required for advantage estimation"
        ds = ds.add_column(
            "advantage",
            estimate_advantage(ds, index_cfg.data),
            new_fingerprint="advantage",  # type: ignore
        )

    query_ds = get_query_data(index_cfg, query_cfg)
    print("Fetched query")

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, index_cfg, query_cfg, ds, query_ds)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = start_processes(
            "query",
            dist_worker,
            args={
                i: (i, world_size, index_cfg, query_cfg, ds, query_ds)
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

    os.rename(index_cfg.partial_run_path, index_cfg.run_path)
