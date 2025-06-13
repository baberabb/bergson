import json
import random
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, Features, Sequence, Value
from tqdm.auto import tqdm, trange
from transformers import PreTrainedModel

from .data import MemmapDataset, pad_and_tensor
from .gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientCollector,
    GradientProcessor,
    Normalizer,
)


def build_index(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[slice] | None = None,
    target_modules: set[str] | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    collector = GradientCollector(
        model.base_model,
        processor,
        target_modules=target_modules,
    )
    features = (
        data.features.copy()
        if isinstance(data, Dataset)
        else Features(input_ids=Value("string"))
    )
    # Make sure gradients are stored in fp16 for efficiency
    features.update(
        gradient=Sequence(Value("float16"), length=collector.gradient_size()),
        loss=Sequence(Value("float16")),
    )

    def generator():
        pbar = tqdm(batches, disable=rank != 0, desc="Building index")
        for sl in pbar:
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

                mask = y[:, 1:] != -100
                denoms = mask.sum(dim=1, dtype=logits.dtype)
                avg_loss = losses.sum(1).div(denoms).mean()
                avg_loss.backward()

                pbar.set_postfix(
                    loss=f"{avg_loss.item():.3f}",
                )
                model.zero_grad()

            gradient = collector.flattened_grads().half().cpu().numpy()
            losses = losses.detach().half().cpu().numpy()

            for i, (g, l, m) in enumerate(zip(gradient, losses, mask.cpu())):
                row = {k: batch[k][i] for k in batch.keys()}
                row.update(gradient=g, loss=l[m])
                yield row

    index = Dataset.from_generator(generator, features)
    index.save_to_disk(path + f"/rank_{rank}.idx")  # type: ignore

    if rank == 0:
        with open(path + "/shapes.json", "w") as f:
            json.dump(collector.shapes(), f, indent=2)

    return index


def fit_normalizers(
    model: PreTrainedModel,
    data: Dataset | MemmapDataset,
    *,
    batches: list[slice] | None = None,
    kind: Literal["adafactor", "adam"] = "adafactor",
    max_documents: int | None = None,
    target_modules: set[str] | None = None,
) -> dict[str, Normalizer]:
    """
    Estimate the second moments of the model's gradients using a subset of the dataset.
    """
    normalizers: dict[str, Normalizer] = {}
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    # If max_tokens is specified, randomly select a subset of batches
    elif max_documents is not None:
        batches = batches.copy()

        rng = random.Random(rank)
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

    N = 0
    callback = adafactor_update if kind == "adafactor" else adam_update
    total = (max_documents or len(batches)) // world_size
    pbar = trange(total, disable=rank != 0, desc="Estimating normalizers")

    for sl in batches:
        batch = data[sl]

        # Update progress
        n = len(batch["input_ids"])
        pbar.update(n)

        N += n
        if total and N >= total:
            break

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
            normalizer.avg_sq.div_(N)

            if dist.is_initialized():
                dist.all_reduce(normalizer.avg_sq, op=dist.ReduceOp.AVG)

        elif isinstance(normalizer, AdafactorNormalizer):
            normalizer.row.div_(N)
            normalizer.col.div_(N)

            if dist.is_initialized():
                dist.all_reduce(normalizer.row, op=dist.ReduceOp.AVG)
                dist.all_reduce(normalizer.col, op=dist.ReduceOp.AVG)

    return normalizers
