import math
import random
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, Value
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .data import create_index, pad_and_tensor
from .gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientCollector,
    GradientProcessor,
    Normalizer,
)


def collect_gradients(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[slice] | None = None,
    skip_preconditioners: bool = False,
    target_modules: set[str] | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    # Mutable state for the GradientCollector callback
    mod_grads = []
    preconditioners = {}

    # TODO: Handle this more elegantly
    lo = torch.finfo(torch.float16).min
    hi = torch.finfo(torch.float16).max

    def callback(name: str, g: torch.Tensor):
        # Prevent infs when casting to fp16 from bf16 or fp32
        g = g.flatten(1).clamp_(lo, hi)

        # Asynchronously move the gradient to CPU and convert to fp16
        mod_grads.append(g.to(device="cpu", dtype=torch.float16, non_blocking=True))

        # Compute the outer product of the flattened gradient
        if not skip_preconditioners:
            g = g.float()
            preconditioner = preconditioners.get(name, None)
            if preconditioner is None:
                preconditioners[name] = g.mT @ g
            else:
                preconditioner.addmm_(g.mT, g)

    collector = GradientCollector(
        model.base_model,
        callback,
        processor,
        target_modules=target_modules,
    )
    # Allocate space ahead of time for the gradients
    grad_size = sum(math.prod(s) for s in collector.shapes().values())
    grad_buffer = create_index(
        path,
        dtype=np.float16,
        shape=(len(data), grad_size),
    )
    per_doc_losses = torch.full(
        (len(data),),
        device=model.device,
        dtype=torch.float16,
        fill_value=0.0,
    )

    for sl in tqdm(batches, disable=rank != 0, desc="Building index"):
        batch = data[sl]
        x, y = pad_and_tensor(
            batch["input_ids"],  # type: ignore
            labels=batch.get("labels"),  # type: ignore
            device=model.device,
        )

        with collector:
            logits = model(x).logits
            losses = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                y[:, 1:].flatten(),
                reduction="none",
            ).reshape_as(y[:, 1:])

            masks = y[:, 1:] != -100
            denoms = masks.sum(dim=1, dtype=logits.dtype)
            losses = losses.sum(1).div(denoms)

            losses.mean().backward()

            model.zero_grad()

        # This forces a host-device sync, but hopefully the transfer to CPU is
        # already done since we called to("cpu", non_blocking=True) in the callback.
        # We could make this even better, potentially, by using a ring buffer to wait
        # longer before syncing.
        indices = batch.get("_row") or sl
        grad_buffer[indices, :] = torch.cat(mod_grads, dim=1).numpy()
        per_doc_losses[indices] = losses.detach().type_as(per_doc_losses)
        mod_grads.clear()

    chols = {}
    for name, prec in preconditioners.items():
        if dist.is_initialized():
            dist.all_reduce(prec)

        chols[name] = torch.linalg.cholesky(prec / len(data))

    processor.preconditioners = chols
    if dist.is_initialized():
        dist.reduce(per_doc_losses, dst=0)

    if rank == 0:
        data = data.sort("_row").remove_columns("_row")
        data = data.add_column(
            "loss",
            per_doc_losses.cpu().numpy(),
            feature=Value("float16"),
            new_fingerprint="loss",
        )
        data.save_to_disk(path + "/data.hf")

        processor.save(path)

    # Make sure the gradients are written to disk
    grad_buffer.flush()


def fit_normalizers(
    model: PreTrainedModel,
    data: Dataset,
    *,
    kind: Literal["adafactor", "adam"] = "adafactor",
    max_documents: int | None = None,
    target_modules: set[str] | None = None,
) -> dict[str, Normalizer]:
    """
    Estimate the second moments of the model's gradients using a subset of the dataset.
    """
    max_documents = max_documents or len(data)
    normalizers: dict[str, Normalizer] = {}
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Round down to nearest multiple of world_size
    max_documents -= max_documents % world_size

    batches = [
        slice(idx + rank, idx + rank + 1) for idx in range(0, max_documents, world_size)
    ]
    # Just to make the pbar more accurate
    rng = random.Random(0)
    rng.shuffle(batches)

    def adafactor_update(name: str, g: torch.Tensor):
        # We follow the tensor2tensor implementation of Adafactor, which
        # takes the mean rather than summing over the rows and columns.
        # row: mean over columns, shape [O]
        sq = g.float().square_().sum(0)
        row_acc = sq.mean(dim=1)
        # col: mean over rows,    shape [I]
        col_acc = sq.mean(dim=0)

        if (normalizer := normalizers.get(name)) is None:
            # initialize accumulators at zero
            normalizers[name] = normalizer = AdafactorNormalizer(
                torch.zeros_like(row_acc),
                torch.zeros_like(col_acc),
            )
        else:
            assert isinstance(normalizer, AdafactorNormalizer)

        # in‐place accumulate
        normalizer.row.add_(row_acc)
        normalizer.col.add_(col_acc)

    def adam_update(name: str, g: torch.Tensor):
        sq = g.square_().float().sum(0)

        # initialize accumulators at zero
        if (normalizer := normalizers.get(name)) is None:
            normalizers[name] = normalizer = AdamNormalizer(torch.zeros_like(sq))
        else:
            assert isinstance(normalizer, AdamNormalizer)

        # in‐place accumulate
        normalizer.avg_sq.add_(sq)

    callback = adafactor_update if kind == "adafactor" else adam_update

    for sl in tqdm(batches, disable=rank != 0, desc="Estimating normalizers"):
        batch = data[sl]

        with GradientCollector(
            model.base_model,
            closure=callback,
            target_modules=target_modules,
        ):
            x, y = pad_and_tensor(
                batch["input_ids"],  # type: ignore
                labels=batch.get("labels", None),  # type: ignore
                device=model.device,
            )
            model(x, labels=y).loss.backward()
            model.zero_grad()

    # Divide by the number of documents processed and average across all ranks
    for normalizer in normalizers.values():
        if isinstance(normalizer, AdamNormalizer):
            normalizer.avg_sq.div_(max_documents)

            if dist.is_initialized():
                dist.all_reduce(normalizer.avg_sq, op=dist.ReduceOp.AVG)

        elif isinstance(normalizer, AdafactorNormalizer):
            normalizer.row.div_(max_documents)
            normalizer.col.div_(max_documents)

            if dist.is_initialized():
                dist.all_reduce(normalizer.row, op=dist.ReduceOp.AVG)
                dist.all_reduce(normalizer.col, op=dist.ReduceOp.AVG)

    return normalizers
