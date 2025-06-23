import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from bergson.data import pad_and_tensor
from bergson.gradients import (
    GradientCollector,
    GradientProcessor,
)


def compute_covariance(
    model: PreTrainedModel,
    data: Dataset,
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

    if rank == 0:
        print(f"Computing covariance matrices for {len(data)} documents...")
    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    # Mutable state for the GradientCollector callback
    mod_grads = []
    activation_covariances = {}

    gradient_covariances = {}

    # TODO: Handle this more elegantly
    lo = torch.finfo(torch.float16).min
    hi = torch.finfo(torch.float16).max

    def callback_activation(name: str, a: torch.Tensor):
        activation_covariance = activation_covariances.get(name, None)

        # a = a.clamp_(lo, hi)
        a = a.reshape(-1, a.shape[-1])  # [N*S, O]

        if activation_covariance is None:
            activation_covariances[name] = a.mT.matmul(a)
        else:
            activation_covariance.addmm_(a.mT, a)

    def callback_gradient(name: str, g: torch.Tensor):
        gradient_covariance = gradient_covariances.get(name, None)

        # Prevent infs when casting to fp16 from bf16 or fp32
        g = g.reshape(-1, g.shape[-1])  # [N*S, O]
        # g = g.clamp_(lo, hi)  # [N*S, O]

        if gradient_covariance is None:
            # Initialize the covariance matrix for this module
            gradient_covariances[name] = g.T.matmul(g)
        else:
            gradient_covariances[name].addmm_(g.T, g)  # [O,O]

    collector = GradientCollector(
        model.base_model,
        closure=callback_gradient,
        processor=processor,
        target_modules=target_modules,
        fwd_closure=callback_activation,
    )

    # per_doc_losses = torch.full(
    #     (len(data),),
    #     device=model.device,
    #     dtype=torch.float16,
    #     fill_value=0.0,
    # )

    for sl in tqdm(batches, disable=rank != 0, desc="Computing covariances"):
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
        # indices = batch.get("_row") or sl
        # per_doc_losses[indices] = losses.detach().type_as(per_doc_losses)
        mod_grads.clear()

        if dist.is_initialized():
            for activation_covariance in activation_covariances.values():
                dist.all_reduce(activation_covariance, op=dist.ReduceOp.SUM)
            for gradient_covariance in gradient_covariances.values():
                dist.all_reduce(gradient_covariance, op=dist.ReduceOp.SUM)

    # if dist.is_initialized():
    #     dist.reduce(per_doc_losses, dst=0)

    if rank == 0:
        save_file(activation_covariances, os.path.join(path, "activation_covariance.safetensors"))
        save_file(gradient_covariances, os.path.join(path, "gradient_covariance.safetensors"))

        print(f"Covariance matrices saved to {path}.")


def compute_eigendecomposition(path: str):
    covariances = load_file(path)

    base, ext = os.path.splitext(path)
    eigen_path = base + "_eigen" + ext
    covariances_eigen = {}
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"Computing covariance eigendecompositions for {os.path.basename(path)}...")

    for name, covariance in covariances.items():
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.to(torch.float32))

        covariances_eigen[name] = eigenvectors.to(torch.float16).contiguous()

    save_file(
        covariances_eigen,
        eigen_path,
    )

    print(f"Eigendecomposition saved to {eigen_path}.")


def compute_eigenvalue_correction(
    model: PreTrainedModel,
    data: Dataset,
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

    if rank == 0:
        print(f"Computing eigenvalue correction for {len(data)} documents...")
    # Batch size of one by default
    if batches is None:
        batches = [slice(idx, idx + 1) for idx in range(len(data))]

    activation_covariance_eigen = load_file(path + "/activation_covariance_eigen.safetensors", device=f"cuda:{rank}")
    gradient_covariance_eigen = load_file(path + "/gradient_covariance_eigen.safetensors", device=f"cuda:{rank}")

    # Mutable state for the GradientCollector callback
    mod_grads = []

    eigenvalue_corrections = {}

    # TODO: Make this faster using Kronecker product structure of g
    def callback_gradient(name: str, g: torch.Tensor):
        eigenvalue_correction = eigenvalue_corrections.get(name, None)
        transformed_g = torch.einsum("N S O, S S-> N S O", g, gradient_covariance_eigen[name])
        transformed_g = torch.einsum("N S O, O O -> N S O", transformed_g, activation_covariance_eigen[name])

        if eigenvalue_correction is None:
            # Initialize the covariance matrix for this module
            eigenvalue_corrections[name] = (transformed_g**2).mean(dim=0)
        else:
            eigenvalue_corrections[name].add((transformed_g**2).mean(dim=0))  # [O,O]

    collector = GradientCollector(
        model.base_model,
        closure=callback_gradient,
        processor=processor,
        target_modules=target_modules,
    )

    # per_doc_losses = torch.full(
    #     (len(data),),
    #     device=model.device,
    #     dtype=torch.float16,
    #     fill_value=0.0,
    # )

    for sl in tqdm(batches, disable=rank != 0, desc="Computing eigenvalue correction"):
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
        # indices = batch.get("_row") or sl
        # per_doc_losses[indices] = losses.detach().type_as(per_doc_losses)
        mod_grads.clear()

        if dist.is_initialized():
            for eigenvalue_correction in eigenvalue_corrections.values():
                dist.all_reduce(eigenvalue_correction, op=dist.ReduceOp.SUM)

    # if dist.is_initialized():
    #     dist.reduce(per_doc_losses, dst=0)

    if rank == 0:
        save_file(eigenvalue_corrections, os.path.join(path, "eigenvalue_corrections.safetensors"))

        print(f"Covariance matrices saved to {os.path.join(path, 'eigenvalue_corrections.safetensors')}.")
