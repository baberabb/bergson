import json
from pathlib import Path
from typing import Literal

import torch
from datasets import Dataset

from .data import QueryConfig, load_gradient_dataset
from .gradients import GradientProcessor


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


def get_query_grads(query_cfg: QueryConfig, device: torch.device, dtype: torch.dtype):
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

    query_ds.set_format("torch", columns=query_cfg.modules)

    query_grads = preprocess_grads(
        query_ds,
        query_cfg.modules,
        query_cfg.unit_normalize,
        query_cfg.batch_size,
        device,
        dtype,
        accumulate_grads="mean" if query_cfg.score == "mean" else "none",
        normalize_accumulated_grad=query_cfg.score == "mean",
    )

    return query_grads
