import os
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from safetensors.torch import load_file, save_file
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from bergson.data import pad_and_tensor
from bergson.gradients import (
    GradientProcessor,
)
from bergson.hessians.collector import EkfacCollector


def sharded_covariance(
    target_info: dict, activation_covariances: dict, gradient_covariances: dict, model: PreTrainedModel
):
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    for name, (device, weight_shape) in target_info.items():
        # Activation covariance A^T A has shape [in_dim, in_dim]
        in_dim = weight_shape[1]
        if in_dim % world_size != 0:
            raise ValueError(f"Activation dim {in_dim} for {name} not divisible by world_size {world_size}")
        shard_size = in_dim // world_size
        activation_covariances[name] = torch.zeros((shard_size, in_dim), device=model.device, dtype=torch.float16)

        # Gradient covariance G^T G has shape [out_dim, out_dim]
        out_dim = weight_shape[0]
        if out_dim % world_size != 0:
            raise ValueError(f"Gradient dim {out_dim} for {name} not divisible by world_size {world_size}")
        shard_size = out_dim // world_size
        gradient_covariances[name] = torch.zeros((shard_size, out_dim), device=model.device, dtype=torch.float16)


def compute_covariance(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[list[int]] | None = None,
    target_modules: set[str] | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Set up the TensorBoard trace handler
    # It will save a trace file for each rank in the specified directory
    trace_handler = tensorboard_trace_handler(dir_name="profiler_logs", worker_name=f"rank_{rank}", use_gzip=True)

    if rank == 0:
        print(f"Computing covariance matrices for {len(data)} documents...")
        log_dir = "profiler_logs"
        os.makedirs(log_dir, exist_ok=True)

    activation_covariances = {}
    gradient_covariances = {}

    collector_for_shapes = EkfacCollector(model.base_model, target_modules=target_modules)
    target_info = collector_for_shapes.target_info
    del collector_for_shapes

    sharded_covariance(
        target_info=target_info,
        activation_covariances=activation_covariances,
        gradient_covariances=gradient_covariances,
        model=model,
    )

    # def callback_activation(name: str, a: torch.Tensor):
    #     activation_covariance = activation_covariances.get(name, None)

    #     # a = a.clamp_(lo, hi)
    #     a = a.reshape(-1, a.shape[-1])  # [N*S, O]

    #     if activation_covariance is None:
    #         activation_covariances[name] = a.mT.matmul(a)
    #     else:
    #         activation_covariance.to(f"cuda:{rank}")
    #         activation_covariance.addmm_(a.mT, a)

    def callback_activation(name: str, a: torch.Tensor):
        sharded_cov_matrix = activation_covariances[name]  # Our stored slice
        # a = a.clamp_(lo, hi)
        a = a.reshape(-1, a.shape[-1])  # [N*S, O]
        local_update = a.mT @ a

        dist.all_reduce(local_update, op=dist.ReduceOp.SUM)

        start_row = rank * sharded_cov_matrix.shape[0]
        end_row = (rank + 1) * sharded_cov_matrix.shape[0]
        update_slice = local_update[start_row:end_row, :]

        # Add it to our permanently stored slice
        sharded_cov_matrix.add_(update_slice)

    def callback_gradient(name: str, g: torch.Tensor):
        gradient_covariance = gradient_covariances[name]

        g = g.reshape(-1, g.shape[-1])  # [N*S, O]
        local_update = g.mT @ g
        dist.all_reduce(local_update, op=dist.ReduceOp.SUM)

        start_row = rank * gradient_covariance.shape[0]
        end_row = (rank + 1) * gradient_covariance.shape[0]
        update_slice = local_update[start_row:end_row, :]
        # Add it to our permanently stored slice
        gradient_covariance.add_(update_slice)

    collector = EkfacCollector(
        model.base_model,
        closure=callback_gradient,
        processor=processor,
        target_modules=target_modules,
        fwd_closure=callback_activation,
    )

    total_processed = torch.tensor(0, device=model.device)
    step = 0
    # The profiler context manager
    my_schedule = schedule(wait=0, warmup=0, active=1, repeat=1)
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Profile both
        on_trace_ready=trace_handler,
        schedule=my_schedule,  # <-- USE THE SCHEDULE
        record_shapes=True,
        with_stack=True,  # Catches the Python call stack, very useful but adds overhead
        profile_memory=True,  # THIS IS THE KEY for memory tracking
    )

    with prof:
        for sl in tqdm(batches, disable=rank != 0, desc="Computing covariances"):
            batch = data[sl]
            x, y = pad_and_tensor(
                batch["input_ids"],  # type: ignore
                labels=batch.get("labels"),  # type: ignore
                device=model.device,
            )

            total_processed += x.numel()

            with record_function(f"step_{step}"):
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

                # if dist.is_initialized():
                #     for activation_covariance in activation_covariances.values():
                #         dist.all_reduce(activation_covariance, op=dist.ReduceOp.SUM)

                #     for gradient_covariance in gradient_covariances.values():
                #         dist.all_reduce(gradient_covariance, op=dist.ReduceOp.SUM)

            prof.step()

    if dist.is_initialized():
        dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)

    if rank == 0:
        save_file(activation_covariances, os.path.join(path, "activation_covariance.safetensors"))
        save_file(gradient_covariances, os.path.join(path, "gradient_covariance.safetensors"))
        save_file({"total_processed": total_processed}, os.path.join(path, "total_processed.safetensors"))

        print(f"Covariance matrices saved to {path}.")


def compute_eigendecomposition(path: str, type: Literal["activation", "gradient"]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    covariances = load_file(os.path.join(path, f"{type}_covariance.safetensors"), device=device)

    num_processed = load_file(os.path.join(path, "total_processed.safetensors"), device=device)["total_processed"]

    eigen_path = path + f"/{type}_covariance_eigen.safetensors"
    covariances_eigen = {}

    for name, covariance in covariances.items():
        original_dtype = covariance.dtype

        covariance_normalized = covariance.to(torch.float32) / num_processed

        # Check for and replace NaN/Inf values
        if torch.isnan(covariance_normalized).any() or torch.isinf(covariance_normalized).any():
            print("Warning: Found NaN/Inf values in covariance matrix")
            # Replace NaN and Inf with 0
            covariance_normalized = torch.nan_to_num(covariance_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_normalized)
        except:
            eigenvectors = covariance_normalized
        covariances_eigen[name] = eigenvectors.to(original_dtype).contiguous()

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
    batches: list[list[int]] | None = None,
    target_modules: set[str] | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        print(f"Computing eigenvalue correction for {len(data)} documents...")

    activation_covariance_eigen = load_file(path + "/activation_covariance_eigen.safetensors", device=f"cuda:{rank}")
    gradient_covariance_eigen = load_file(path + "/gradient_covariance_eigen.safetensors", device=f"cuda:{rank}")

    eigenvalue_corrections = {}

    # TODO: Make this faster using Kronecker product structure of g
    def callback_gradient(name: str, g: torch.Tensor):
        eigenvalue_correction = eigenvalue_corrections.get(name, None)

        right_transformed_g = torch.einsum("N S O, S S-> N S O", g, gradient_covariance_eigen[name])

        left_transformed_g = torch.einsum("N S O, O O -> N S O", right_transformed_g, activation_covariance_eigen[name])

        if eigenvalue_correction is None:
            # Initialize the covariance matrix for this module
            eigenvalue_corrections[name] = (left_transformed_g**2).mean(dim=0)
        else:
            eigenvalue_corrections[name].add((left_transformed_g**2).mean(dim=0))  # [O,O]

    collector = EkfacCollector(
        model.base_model,
        closure=callback_gradient,
        processor=processor,
        target_modules=target_modules,
    )

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

        if dist.is_initialized():
            for eigenvalue_correction in eigenvalue_corrections.values():
                dist.all_reduce(eigenvalue_correction, op=dist.ReduceOp.SUM)

    if rank == 0:
        save_file(eigenvalue_corrections, os.path.join(path, "eigenvalue_corrections.safetensors"))

        print(f"Covariance matrices saved to {os.path.join(path, 'eigenvalue_corrections.safetensors')}.")
