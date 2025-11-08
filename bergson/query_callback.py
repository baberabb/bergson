import json
import os
from pathlib import Path
from typing import Callable, Literal

import torch
from datasets import Dataset

from .data import QueryConfig, load_gradient_dataset
from .gradients import GradientProcessor

# Do we want to map from HF ds to HF ds or memmap to memmap?
# We need info.json, so let's do HF to HF.


# LESS does mean gradient for task -> unit normalization
# And then they sum the result across checkpoints (one mean grad per checkpoint)
def transform_gradients(
    source_path: Path,
    destination_path: Path,
    unit_normalize_individual_grads: bool,
    normalize_accumulated_grad: bool,
    accumulate_grads: Literal["mean", "sum", "none"] = "none",
    batch_size: int = 1024,
):
    pass

    # Load the source dataset
    source_ds = load_gradient_dataset(source_path)
    with open(os.path.join(source_path, "info.json"), "r") as f:
        source_modules = json.load(f)["dtype"]["names"]
    source_ds = source_ds.with_format("torch", columns=source_modules)

    # Load the destination dataset
    destination_ds = load_gradient_dataset(destination_path)
    destination_ds = destination_ds.with_format(
        "torch", columns=source_modules, output_all_columns=True
    )

    # Transform the gradients
    destination_ds = destination_ds.map(
        transform_gradients, batched=True, batch_size=batch_size
    )


class Scorer:
    num_scores: int

    def __init__(self, callback: Callable, num_scores: int):
        self.callback = callback
        self.num_scores = num_scores

    def __call__(self, mod_grads: dict[str, torch.Tensor], **kwargs):
        return self.callback(mod_grads, **kwargs)


def get_query_data(query_cfg: QueryConfig):
    """
    Load and optionally precondition the query dataset. Preconditioners
    may be mixed as described in https://arxiv.org/html/2410.17413v1#S3.
    """
    # Collect the query gradients if they don't exist
    if not os.path.exists(query_cfg.query_path):
        raise FileNotFoundError(
            f"Query dataset not found at {query_cfg.query_path}. "
            "Please build a query dataset index first."
        )

    # Load the query dataset
    with open(os.path.join(query_cfg.query_path, "info.json"), "r") as f:
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


def batched_mean(
    grad_ds: Dataset,
    grad_column_names: list[str],
    unit_normalize: bool,
    batch_size: int,
    device: torch.device,
    final_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Compute the mean of the gradients in the dataset. Returns a dictionary
    of mean gradients with shape [1, grad_dim]."""
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

    ss_acc = ss_acc.sqrt()
    assert ss_acc > 0, "Sum of squares of entire dataset is zero"

    grad_ds.map(
        sum_,
        batched=True,
        batch_size=batch_size,
    )

    return {
        column_name: (acc[column_name] / ss_acc / len(grad_ds))
        .unsqueeze(0)
        .to(final_dtype)
        for column_name in grad_column_names
    }


# TODO these should all do inner products and we should handle normalization
# in the scorer writer maybe
# Like just have modules wise inner products and full gradient inner product
# and build everything around that.
def get_module_wise_mean_scorer(
    query_ds: Dataset,
    query_cfg: QueryConfig,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Compute the mean query and return a callback function that scores gradients
    according to their inner products or cosine similarities with the mean query.
    """
    callback_query = batched_mean(
        query_ds,
        query_cfg.modules,
        query_cfg.unit_normalize,
        query_cfg.batch_size,
        device,
        final_dtype=dtype,
    )

    @torch.inference_mode()
    def callback(mod_grads: dict[str, torch.Tensor], name: str):
        module_scores = mod_grads[name] @ callback_query[name].T
        module_scores = module_scores.to("cpu", non_blocking=True)

        # We can't normalize the module gradient because we don't have the full gradient
        # norm so we accumulate the sum of squares and normalize later.
        # [num_items, grad_dim]
        mod_grads[name].pow_(2)
        sum_of_squares = mod_grads[name].sum(dim=1).to("cpu", non_blocking=True)

        return module_scores, sum_of_squares

    return Scorer(callback, num_scores=1)


# TODO do we normalize then mean then normalize again
# or mean then normalize
# LESS has the sum of cosine similarities...


def get_mean_scorer(
    query_ds: Dataset, query_cfg: QueryConfig, device: torch.device, dtype: torch.dtype
):
    """
    Compute the mean query and return a callback function that scores gradients
    according to their inner products or cosine similarities with the mean query.
    """
    callback_query = batched_mean(
        query_ds,
        query_cfg.modules,
        query_cfg.unit_normalize,
        query_cfg.batch_size,
        device,
        final_dtype=dtype,
    )

    callback_query = torch.cat(
        [
            callback_query[module].to(device=device, dtype=dtype)
            for module in query_cfg.modules
        ],
        dim=1,
    )

    @torch.inference_mode()
    def callback(mod_grads: dict[str, torch.Tensor]):
        grads = torch.cat([mod_grads[name] for name in query_cfg.modules], dim=1)

        if query_cfg.unit_normalize:
            grads /= grads.norm(dim=1, keepdim=True)
        return grads @ callback_query.T

    return Scorer(callback, num_scores=1)


def get_scorer(
    query_ds: Dataset, query_cfg: QueryConfig, device: torch.device, dtype: torch.dtype
):
    """
    Compute the mean query and return a callback function that scores gradients
    according to their inner products or cosine similarities with the mean query.
    """
    callback_query = torch.cat(
        [
            query_ds[:][module].to(device=device, dtype=dtype)
            for module in query_cfg.modules
        ],
        dim=1,
    )
    if query_cfg.unit_normalize:
        callback_query /= callback_query.norm(dim=1, keepdim=True)

    @torch.inference_mode()
    def callback(mod_grads: dict[str, torch.Tensor]):
        grads = torch.cat([mod_grads[name] for name in query_cfg.modules], dim=1)

        if query_cfg.unit_normalize:
            grads /= grads.norm(dim=1, keepdim=True)
        return grads @ callback_query

    return Scorer(callback, num_scores=len(query_ds))


def get_nearest_query(
    query_ds: Dataset, query_cfg: QueryConfig, device: torch.device, dtype: torch.dtype
):
    """
    Return a callback function that scores gradients according to their cosine
    similarities or inner products with the most similar gradient in the query
    dataset.
    """

    queries = torch.cat([query_ds[:][name] for name in query_cfg.modules], dim=1).to(
        device=device, dtype=dtype
    )

    if query_cfg.unit_normalize:
        queries /= queries.norm(dim=1, keepdim=True)

    def callback(mod_grads: dict[str, torch.Tensor]):
        grads = torch.cat([mod_grads[name] for name in query_cfg.modules], dim=1)
        if query_cfg.unit_normalize:
            grads /= grads.norm(dim=1, keepdim=True)

        # Calculate scores as the max of the inner products with the queries
        all_scores = grads @ queries.T
        return all_scores.max(dim=-1).values

    return Scorer(callback, num_scores=1)


def get_scorer_callback(
    query_cfg: QueryConfig, module_wise: bool, device: torch.device, dtype: torch.dtype
) -> Scorer:
    query_ds = get_query_data(query_cfg)

    if query_cfg.score == "mean":
        if module_wise:
            return get_module_wise_mean_scorer(query_ds, query_cfg, device, dtype)
        else:
            return get_mean_scorer(query_ds, query_cfg, device, dtype)
    elif query_cfg.score == "nearest":
        assert not module_wise, "Nearest query is not supported for module-wise scoring"
        return get_nearest_query(query_ds, query_cfg, device, dtype)
    elif query_cfg.score == "individual":
        assert (
            not module_wise
        ), "Individual scoring is not supported for module-wise scoring"
        return get_scorer(query_ds, query_cfg, device, dtype)
    else:
        raise ValueError(f"Invalid scoring method: {query_cfg.score}")
