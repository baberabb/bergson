import json
import os
import shutil
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

import torch
import torch.distributed as dist
from datasets import Dataset, IterableDataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .collection import collect_gradients
from .data import (
    IndexConfig,
    QueryConfig,
    allocate_batches,
    load_gradient_dataset,
    load_gradients,
)
from .gradients import GradientProcessor
from .launch import launch_distributed_run
from .score_writer import MemmapScoreWriter
from .scorer import get_scorer
from .utils import assert_type
from .worker_utils import create_processor, setup_data_pipeline, setup_model_and_peft


def preprocess_grads(
    grad_ds: Dataset,
    grad_column_names: list[str],
    unit_normalize: bool,
    batch_size: int,
    device: torch.device,
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
            column_name: grad_ds[:][column_name].to(device=device)
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
            column_name: (acc[column_name] / ss_acc / len(grad_ds)).unsqueeze(0)
            for column_name in grad_column_names
        }
    elif accumulate_grads == "sum":
        grads = {
            column_name: (acc[column_name] / ss_acc).unsqueeze(0)
            for column_name in grad_column_names
        }
    elif accumulate_grads == "none":
        grads = {
            column_name: grad_ds[:][column_name].to(device=device) / ss_acc
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


def get_query_ds(query_cfg: QueryConfig, device: str, rank: int | None = None):
    """
    Load and preprocess the query dataset to get the query gradients. Preconditioners
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

    try:
        query_ds = load_gradient_dataset(Path(query_cfg.query_path), structured=True)
    except ValueError as e:
        if "integer won't fit into a C int" not in str(e):
            raise e

        if rank == 0 or rank is None:
            print(
                "Query gradients are too large to load with structure. "
                "Attempting to load without structure..."
            )

        mmap = load_gradients(Path(query_cfg.query_path), structured=False)

        # Convert unstructured gradients to a dictionary of module-wise tensors
        with open(query_path / "info.json", "r") as f:
            metadata = json.load(f)
            grad_sizes = metadata["grad_sizes"]

        sizes = torch.tensor(list(grad_sizes.values()))
        module_offsets = torch.tensor([0] + torch.cumsum(sizes, dim=0).tolist())

        query_ds = Dataset.from_dict(
            {
                name: mmap[:, module_offsets[i] : module_offsets[i + 1]].copy()
                for i, name in enumerate(grad_sizes.keys())
                if name in target_modules
            }
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
                map_location=device,
            ).preconditioners
        if use_i:
            assert query_cfg.index_preconditioner_path is not None
            i = GradientProcessor.load(
                Path(query_cfg.index_preconditioner_path), map_location=device
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
        mixed_preconditioner = {
            k: v.to(device) for k, v in mixed_preconditioner.items()
        }

        def precondition(batch):
            for name in target_modules:
                batch[name] = (
                    batch[name].to(device) @ mixed_preconditioner[name]
                ).cpu()

            return batch

        query_ds = query_ds.map(
            precondition, batched=True, batch_size=query_cfg.batch_size
        )

    return query_ds.with_format("torch", columns=query_cfg.modules)


def query_worker(
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

    model, target_modules = setup_model_and_peft(index_cfg, rank)
    model = cast(PreTrainedModel, model)
    processor = create_processor(index_cfg, rank)

    attention_cfgs = {
        module: index_cfg.attention for module in index_cfg.split_attention_modules
    }

    query_ds = get_query_ds(query_cfg, f"cuda:{rank}", rank)
    query_grads = preprocess_grads(
        query_ds,
        query_cfg.modules,
        query_cfg.unit_normalize,
        query_cfg.batch_size,
        torch.device(f"cuda:{rank}"),
        accumulate_grads="mean" if query_cfg.score == "mean" else "none",
        normalize_accumulated_grad=query_cfg.score == "mean",
    )

    num_scores = len(query_grads[query_cfg.modules[0]])

    kwargs = {
        "model": model,
        "data": ds,
        "processor": processor,
        "cfg": index_cfg,
        "target_modules": target_modules,
        "attention_cfgs": attention_cfgs,
    }

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"], index_cfg.token_batch_size)
        kwargs["batches"] = batches

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
            model.dtype if model.dtype != "auto" else torch.float32,
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
            batches = allocate_batches(
                ds_shard["length"][:], index_cfg.token_batch_size
            )
            kwargs["ds"] = ds_shard
            kwargs["batches"] = batches

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
                model.dtype if model.dtype != "auto" else torch.float32,
            )
            kwargs["scorer"] = scorer

            collect_gradients(**kwargs)

            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Collecting gradients"):
            buf.append(ex)
            if len(buf) == index_cfg.stream_shard_size:
                flush(kwargs=kwargs)

        flush(kwargs=kwargs)  # Final flush
        if rank == 0:
            processor.save(index_cfg.partial_run_path)


def query(cfg: IndexConfig, query_cfg: QueryConfig):
    cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    with (cfg.partial_run_path / "index_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    ds = setup_data_pipeline(cfg)

    launch_distributed_run("query", query_worker, [cfg, query_cfg, ds])

    shutil.move(cfg.partial_run_path, cfg.run_path)
