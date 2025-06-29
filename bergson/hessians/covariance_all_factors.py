import gc
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


def all_gather_matrices(local_matrix):
    """
    Gather matrices from all processes and concatenate them
    """
    world_size = dist.get_world_size()

    # Gather all matrices
    gathered_matrices = [torch.zeros_like(local_matrix) for _ in range(world_size)]
    dist.all_gather(gathered_matrices, local_matrix)

    # Concatenate along the first dimension (typical sharding dimension)
    full_matrix = torch.cat(gathered_matrices, dim=0)

    return full_matrix


def gather_and_save_covariances(local_covariances_dict: dict[str, torch.Tensor], path: str):
    """
    Gather matrices from all processes and concatenate them
    """
    if not dist.is_initialized():
        torch.save(local_covariances_dict, path)
        del local_covariances_dict
        return

    gathered_dict = {}

    rank = dist.get_rank()

    for name in local_covariances_dict:
        gathered_tensor = all_gather_matrices(local_covariances_dict[name])
        if rank == 0:
            gathered_dict[name] = gathered_tensor
        else:
            del gathered_tensor  # Only keep the tensor on rank 0 to save memory

    if rank == 0:
        save_file(gathered_dict, path)
        print(f"Covariance matrices saved to {path}.")

    del local_covariances_dict


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
    my_schedule = schedule(wait=0, warmup=0, active=4, repeat=1)
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

            prof.step()

    if dist.is_initialized():
        dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)
    if rank == 0:
        save_file({"total_processed": total_processed}, os.path.join(path, "total_processed.safetensors"))

    gather_and_save_covariances(
        local_covariances_dict=activation_covariances, path=os.path.join(path, "activation_covariance.safetensors")
    )
    torch.cuda.empty_cache()  # Clear cache to avoid OOM issues
    gc.collect()
    gather_and_save_covariances(
        local_covariances_dict=gradient_covariances, path=os.path.join(path, "gradient_covariance.safetensors")
    )
    torch.cuda.empty_cache()  # Clear cache to avoid OOM issues
    gc.collect()


def compute_eigendecomposition(path: str, type: Literal["activation", "gradient"]):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    covariances = load_file(os.path.join(path, f"{type}_covariance.safetensors"), device=device)

    num_processed = load_file(os.path.join(path, "total_processed.safetensors"), device=device)["total_processed"]

    eigen_path = path + f"/{type}_covariance_eigen.safetensors"
    covariances_eigen = {}

    covariance_list = list(covariances.values())

    covariance_list_rank = covariance_list
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
