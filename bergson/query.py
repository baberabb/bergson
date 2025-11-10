import json
import os
import shutil
import socket
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

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

from .build import estimate_advantage
from .collection import collect_gradients
from .data import (
    IndexConfig,
    QueryConfig,
    allocate_batches,
    load_data_string,
    load_gradient_dataset,
    tokenize,
)
from .gradients import GradientProcessor
from .peft import detect_peft_modules
from .score_writer import MemmapScoreWriter
from .scorer import get_scorer
from .utils import assert_type, get_layer_list


def preprocess_grads(
    grad_ds: Dataset,
    grad_column_names: list[str],
    unit_normalize: bool,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    accumulate_grads: Literal["mean", "sum", "none"] = "none",
    normalize_accumulated_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Preprocess the gradients in the dataset. Returns a dictionary
    of preprocessed gradients with shape [1, grad_dim]. Preprocessing
    includes some combination of unit normalization, accumulation,
    accumulated gradient normalization, and dtype conversion."""

    # Short-circuit if possible
    if accumulate_grads == "none" and not unit_normalize:
        return {
            column_name: grad_ds[:][column_name].to(device=device, dtype=dtype)
            for column_name in grad_column_names
        }

    # Get sum and sum of squares of the gradients
    acc = {
        column_name: torch.zeros_like(
            grad_ds[0][column_name], device=device, dtype=torch.float32
        )
        for column_name in grad_column_names
    }
    ss_acc = torch.tensor(0.0, device=device, dtype=torch.float32)
    if not unit_normalize:
        ss_acc.fill_(1.0)

    def sum_(cols):
        nonlocal ss_acc

        for column_name in grad_column_names:
            x = cols[column_name].to(device=device, dtype=torch.float32)
            acc[column_name].add_(x.sum(0))

            if unit_normalize:
                # To normalize the mean gradient we can divide by the sum of
                # squares of every gradient element in the dataset
                ss_acc += x.pow(2).sum()

    grad_ds.map(
        sum_,
        batched=True,
        batch_size=batch_size,
    )

    ss_acc = ss_acc.sqrt()
    assert ss_acc > 0, "Sum of squares of entire dataset is zero"

    # Process the gradient dataset
    if accumulate_grads == "mean":
        grads = {
            column_name: (acc[column_name] / ss_acc / len(grad_ds))
            .unsqueeze(0)
            .to(dtype)
            for column_name in grad_column_names
        }
    elif accumulate_grads == "sum":
        grads = {
            column_name: (acc[column_name] / ss_acc).unsqueeze(0).to(dtype)
            for column_name in grad_column_names
        }
    elif accumulate_grads == "none":
        grads = {
            column_name: grad_ds[:][column_name].to(device=device, dtype=dtype) / ss_acc
            for column_name in grad_column_names
        }
    else:
        raise ValueError(f"Invalid accumulate_grads: {accumulate_grads}")

    # Normalize the accumulated gradient
    if normalize_accumulated_grad:
        grad_norm = torch.cat(
            [grads[column_name].flatten() for column_name in grad_column_names], dim=0
        ).norm()
        for column_name in grad_column_names:
            grads[column_name] /= grad_norm

    return grads


def get_query_ds(query_cfg: QueryConfig):
    """
    Load and optionally precondition the query dataset. Preconditioners
    may be mixed as described in https://arxiv.org/html/2410.17413v1#S3.
    """
    # Collect the query gradients if they don't exist
    query_path = Path(query_cfg.query_path)
    if not query_path.exists():
        raise FileNotFoundError(
            f"Query dataset not found at {query_cfg.query_path}. "
            "Please build a query dataset index first."
        )

    # Load the query dataset
    with open(query_path / "info.json", "r") as f:
        target_modules = json.load(f)["dtype"]["names"]

    if not query_cfg.modules:
        query_cfg.modules = target_modules

    query_ds = load_gradient_dataset(
        Path(query_cfg.query_path), concatenate_gradients=False
    )
    query_ds = query_ds.with_format("torch", columns=target_modules)

    use_q = query_cfg.query_preconditioner_path is not None
    use_i = query_cfg.index_preconditioner_path is not None

    if use_q or use_i:
        q, i = {}, {}
        if use_q:
            assert query_cfg.query_preconditioner_path is not None
            q = GradientProcessor.load(
                Path(query_cfg.query_preconditioner_path),
                map_location="cuda",
            ).preconditioners
        if use_i:
            assert query_cfg.index_preconditioner_path is not None
            i = GradientProcessor.load(
                Path(query_cfg.index_preconditioner_path), map_location="cuda"
            ).preconditioners

        mixed_preconditioner = (
            {
                k: q[k] * query_cfg.mixing_coefficient
                + i[k] * (1 - query_cfg.mixing_coefficient)
                for k in q
            }
            if (q and i)
            else (q or i)
        )
        mixed_preconditioner = {k: v.cuda() for k, v in mixed_preconditioner.items()}

        def precondition(batch):
            for name in target_modules:
                batch[name] = (batch[name].cuda() @ mixed_preconditioner[name]).cpu()

            return batch

        query_ds = query_ds.map(
            precondition, batched=True, batch_size=query_cfg.batch_size
        )

    return query_ds.with_format("torch", columns=query_cfg.modules)


def worker(
    rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    query_cfg: QueryConfig,
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

        model = PeftModel.from_pretrained(  # type: ignore
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

    processor_dir = Path(index_cfg.processor_path or index_cfg.run_path)
    processor_cfg_path = processor_dir / "processor_config.json"

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
            include_bias=index_cfg.include_bias,
        )
        if rank == 0:
            processor.save(index_cfg.partial_run_path)

    if index_cfg.split_attention_modules:
        attention_cfgs = {
            module: index_cfg.attention for module in index_cfg.split_attention_modules
        }
    else:
        attention_cfgs = {}

    # score_writer_dtype = dtype if dtype != "auto" else torch.float32
    score_writer_dtype = torch.float32

    query_ds = get_query_ds(query_cfg)
    query_grads = preprocess_grads(
        query_ds,
        query_cfg.modules,
        query_cfg.unit_normalize,
        query_cfg.batch_size,
        torch.device(f"cuda:{rank}"),
        score_writer_dtype,
        accumulate_grads="mean" if query_cfg.score == "mean" else "none",
        normalize_accumulated_grad=query_cfg.score == "mean",
    )
    num_scores = len(query_grads[query_cfg.modules[0]])

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"][:], index_cfg.token_batch_size)
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
            index_cfg.module_wise,
            torch.device(f"cuda:{rank}"),
            score_writer_dtype,
        )
        collect_gradients(
            model,
            ds,
            processor,
            index_cfg.partial_run_path,
            batches=batches,
            kl_divergence=index_cfg.loss_fn == "kl",
            loss_reduction=index_cfg.loss_reduction,
            skip_preconditioners=index_cfg.skip_preconditioners,
            target_modules=target_modules,
            attention_cfgs=attention_cfgs,
            drop_columns=index_cfg.drop_columns,
            scorer=scorer,
            save_index=index_cfg.save_index,
            module_wise=index_cfg.module_wise,
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
                index_cfg.module_wise,
                torch.device(f"cuda:{rank}"),
                score_writer_dtype,
            )
            collect_gradients(
                model,
                ds_shard,
                processor,
                index_cfg.partial_run_path / f"shard-{shard_id:05d}",
                batches=batches,
                kl_divergence=index_cfg.loss_fn == "kl",
                loss_reduction=index_cfg.loss_reduction,
                skip_preconditioners=index_cfg.skip_preconditioners,
                target_modules=target_modules,
                attention_cfgs=attention_cfgs,
                drop_columns=index_cfg.drop_columns,
                scorer=scorer,
                save_index=index_cfg.save_index,
            )
            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Querying gradients on the fly", disable=rank != 0):
            buf.append(ex)
            if len(buf) == index_cfg.stream_shard_size:
                flush()
        flush()

        if rank == 0:
            processor.save(index_cfg.partial_run_path)


def dist_worker(
    rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    query_cfg: QueryConfig,
    ds: Dataset,
):
    try:
        worker(rank, world_size, index_cfg, query_cfg, ds)
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

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, index_cfg, query_cfg, ds)
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
                i: (i, world_size, index_cfg, query_cfg, ds) for i in range(world_size)
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

    try:
        shutil.move(index_cfg.partial_run_path, index_cfg.run_path)
    except Exception:
        pass
