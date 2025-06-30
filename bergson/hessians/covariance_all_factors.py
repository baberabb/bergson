import gc
import os
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from bergson.data import pad_and_tensor
from bergson.gradients import (
    GradientProcessor,
)
from bergson.hessians.collector import EkfacCollector


class EkfacComputer:
    def __init__(
        self,
        model: PreTrainedModel,
        processor: GradientProcessor,
        data: Dataset,
        path: str,
        *,
        batches: list[list[int]],
        target_modules: set[str] | None = None,
        debug=False,
    ):
        self.model = model
        self.processor = processor
        self.target_modules = target_modules
        self.data = data
        self.batches = batches
        self.path = path
        self.target_info = EkfacCollector(model.base_model, target_modules=target_modules).target_info
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = model.device
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.debug = debug

    def compute_covariance(self):
        """
        Compute projected gradients using a subset of the dataset.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Set up the TensorBoard trace handler
        # It will save a trace file for each rank in the specified directory
        trace_handler = tensorboard_trace_handler(dir_name="profiler_logs", worker_name=f"rank_{rank}", use_gzip=True)

        if rank == 0:
            print(f"Computing covariance matrices for {len(self.data)} documents...")
            log_dir = "profiler_logs"
            os.makedirs(log_dir, exist_ok=True)

        activation_covariances = {}
        gradient_covariances = {}

        self.sharded_covariance(
            activation_covariances=activation_covariances,
            gradient_covariances=gradient_covariances,
        )

        def callback_activation(name: str, a: torch.Tensor):
            sharded_cov_matrix = activation_covariances[name]  # Our stored slice
            # a = a.clamp_(lo, hi)
            a = a.reshape(-1, a.shape[-1])  # [N*S, O]
            local_update = a.mT @ a

            dist.all_reduce(local_update, op=dist.ReduceOp.SUM)

            # Manually sharding
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
            self.model.base_model,
            closure=callback_gradient,
            processor=self.processor,
            target_modules=self.target_modules,
            fwd_closure=callback_activation,
        )

        total_processed = torch.tensor(0, device=self.model.device)
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
            for sl in tqdm(self.batches, disable=rank != 0, desc="Computing covariances"):
                batch = self.data[sl]
                x, y = pad_and_tensor(
                    batch["input_ids"],  # type: ignore
                    labels=batch.get("labels"),  # type: ignore
                    device=self.model.device,
                )

                total_processed += x.numel()

                with record_function(f"step_{step}"):
                    with collector:
                        logits = self.model(x).logits
                        losses = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.size(-1)),
                            y[:, 1:].flatten(),
                            reduction="none",
                        ).reshape_as(y[:, 1:])

                        masks = y[:, 1:] != -100
                        denoms = masks.sum(dim=1, dtype=logits.dtype)
                        losses = losses.sum(1).div(denoms)

                        losses.mean().backward()

                        self.model.zero_grad()

                prof.step()

        if dist.is_initialized():
            dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)
        if rank == 0:
            torch.save(total_processed, os.path.join(self.path, "total_processed.pt"))
            print(f"Total processed: {total_processed.item()}")
        print(f"{self.rank} finished computing gradients.")
        activation_path = os.path.join(self.path, "activation_covariance_sharded")
        gradient_path = os.path.join(self.path, "gradient_covariance_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)
        print(f"{self.rank} about to save gradients.")

        # Save the sharded covariance matrices
        print(f"{self.rank} really about to save gradients.")
        save_file(activation_covariances, os.path.join(activation_path, f"shard_{rank}.safetensors"))
        save_file(gradient_covariances, os.path.join(gradient_path, f"shard_{rank}.safetensors"))
        print(f"{self.rank} finished computing covariance.")

        dist.barrier()
        torch.cuda.empty_cache()
        gc.collect()

    def sharded_covariance(
        self,
        activation_covariances: dict,
        gradient_covariances: dict,
    ):
        for name, (device, weight_shape) in self.target_info.items():
            # Activation covariance A^T A has shape [in_dim, in_dim]
            in_dim = weight_shape[1]
            if in_dim % self.world_size != 0:
                raise ValueError(f"Activation dim {in_dim} for {name} not divisible by world_size {self.world_size}")
            shard_size = in_dim // self.world_size
            activation_covariances[name] = torch.zeros((shard_size, in_dim), device=self.device, dtype=torch.float16)

            # Gradient covariance G^T G has shape [out_dim, out_dim]
            out_dim = weight_shape[0]
            if out_dim % self.world_size != 0:
                raise ValueError(f"Gradient dim {out_dim} for {name} not divisible by world_size {self.world_size}")
            shard_size = out_dim // self.world_size
            gradient_covariances[name] = torch.zeros((shard_size, out_dim), device=self.device, dtype=torch.float16)

    def compute_full_matrix(self, key: str, covariance_type: Literal["activation", "gradient"]):
        """
        Load a full matrix from sharded covariance files.
        """
        shard_path = os.path.join(self.path, f"{covariance_type}_covariance_sharded")
        files = os.listdir(shard_path)
        assert len(files) == self.world_size, f"Expected {self.world_size} shards, found {len(files)} in {self.path}"

        full_matrix = []
        for shard_id in range(self.world_size):
            shard_path_rank = os.path.join(shard_path, f"shard_{shard_id}.safetensors")
            with safe_open(shard_path_rank, framework="pt", device=f"cuda:{self.rank}") as f:
                local_matrix = f.get_tensor(key)

            full_matrix.append(local_matrix)

        full_matrix = torch.cat(full_matrix, dim=0)

        assert full_matrix.shape[0] == full_matrix.shape[1], "Full covariance matrix must be square"

        return full_matrix

    def merge_and_shard_dict(
        self, input_dict: dict[str, torch.Tensor], covariance_type: Literal["activation", "gradient"]
    ):
        for key in self.target_info:
            shard_size = self.target_info[key][1][0] // self.world_size
            d_in, d_out = self.target_info[key][1]
            print(f"{self.rank} Processing {key} with shard size {shard_size}, in_dim {d_in}, out_dim {d_out}.")

            if key not in input_dict:
                d = d_out if covariance_type == "activation" else d_in

                tensor = torch.zeros([d, d], device=self.device, dtype=torch.float16)
            else:
                tensor = input_dict[key]

            dist.barrier()
            print(
                f"{self.rank} Merging and sharding {key} of shape {tensor.shape} with shard size {shard_size} and type {covariance_type}, generally {d_in}x{d_out}."
            )
            dist.barrier()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            # Shard the tensor
            shard = tensor[self.rank * shard_size : (self.rank + 1) * shard_size, :]
            input_dict[key] = shard
        return input_dict

    def compute_eigendecomposition(self, covariance_type: Literal["activation", "gradient"]):
        total_processed = torch.load(os.path.join(self.path, "total_processed.pt"), map_location=f"cuda:{self.rank}")
        target_info_rank = list(self.target_info)[self.rank :: self.world_size]
        print(f"{self.rank} started computing eigenvectors.")
        covariance_eigenvectors = {}

        # activations
        for key in target_info_rank:
            matrix = self.compute_full_matrix(key, covariance_type=covariance_type)
            original_dtype = matrix.dtype
            matrix_normalized = matrix.to(torch.float64) / total_processed
            # # Check for and replace NaN/Inf values
            if torch.isnan(matrix_normalized).any() or torch.isinf(matrix_normalized).any():
                # Replace NaN and Inf with 0
                matrix_normalized = torch.nan_to_num(matrix_normalized, nan=0.0, posinf=0.0, neginf=0.0)
            # try:
            #     eigenvalues, eigenvectors = torch.linalg.eigh(matrix_normalized)
            # except:
            #     eigenvectors = matrix_normalized

            eigenvectors = matrix
            eigenvectors = eigenvectors.to(original_dtype).contiguous()
            covariance_eigenvectors[key] = eigenvectors
        print(
            f"{self.rank} started merging eigenvectors {covariance_eigenvectors.keys()} {[covariance_eigenvectors[k].shape for k in covariance_eigenvectors.keys()]}."
        )
        covariance_eigenvectors = self.merge_and_shard_dict(
            input_dict=covariance_eigenvectors, covariance_type=covariance_type
        )

        eigen_path = os.path.join(self.path, f"{covariance_type}_eigen")

        os.makedirs(eigen_path, exist_ok=True)

        save_file(covariance_eigenvectors, os.path.join(eigen_path, f"shard_{self.rank}.safetensors"))
        gc.collect()
        torch.cuda.empty_cache()

        print(f"{self.rank} Done")


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
