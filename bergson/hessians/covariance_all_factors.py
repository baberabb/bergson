import gc
import os
import random
from contextlib import nullcontext
from typing import Literal, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from jaxtyping import Float
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from bergson.data import pad_and_tensor
from bergson.gradients import (
    GradientProcessor,
)
from bergson.hessians.collector import EkfacCollector


class EkfacComputer:
    """Compute all factors for the EKFAC algorithm as described in https://arxiv.org/abs/2308.03296 Section 2.2.
    For all MLP modules in the model we do the following
    (where we use the notation torch.nn.Linear.weight.shape = [out, in]):

    1. Compute the covariance of the activations A_l (shape [in,in])
    and pseudo-gradients S_{l-1} (shape [out,out]) (Eq. 16).

    2. Compute the eigendecomposition of the covariance matrices Q_A (shape [in,in])
    and Q_S (shape [out,out]) (Eq. 18).

    3. Compute the eigenvalue correction Lambda (shape [out,in]) (Eq.20).

    If using FSDP, we will do everything in a fully sharded manner to save memory.
    As a result, we will save and load shards of tensors to reduce memory usage.
    """

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
        profile=False,
    ):
        self.model = model
        self.processor = processor
        self.target_modules = target_modules  # which modules to compute EKFAC for, by default uses all MLPs
        self.data = data
        self.batches = batches
        self.path = path

        self.target_info = EkfacCollector(model.base_model, target_modules=target_modules).target_info

        self.device = model.device
        self.dtype = model.dtype
        self.debug = debug
        self.profile = profile

        ### FSDP related
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        if self.rank == 0:
            print(f"Computing EKFAC for {list(self.target_info)} target modules.")

    def compute_covariance(self):
        """
        Computes Eq.16 from above reference.
        """

        # These will be sharded
        A_cov_dict = {}
        S_cov_dict = {}

        self._sharded_covariance(
            activation_covariance_dict=A_cov_dict,
            gradient_covariance_dict=S_cov_dict,
            dtype=self.dtype,
        )

        def callback_activation(name: str, a: Float[Tensor, "N S I"]):
            """Forward hook to compute the covariance of activations A_l."""

            A_cov_ki = A_cov_dict[name]  # Our stored slice

            a_bi = a.reshape(-1, a.shape[-1])  # [N*S, i]

            local_update_ii = a_bi.mT @ a_bi

            dist.all_reduce(local_update_ii, op=dist.ReduceOp.SUM)

            # Manually sharding
            start_row = self.rank * A_cov_ki.shape[0]
            end_row = (self.rank + 1) * A_cov_ki.shape[0]
            update_slice_ki = local_update_ii[start_row:end_row, :]

            # Add it to permanently stored slice
            A_cov_ki.add_(update_slice_ki)

        def callback_gradient(name: str, g: Float[Tensor, "N S O"]):
            """Backward hook to compute the covariance of pseudo-gradients S_{l-1}."""
            S_cov_po = S_cov_dict[name]

            g_bo = g.reshape(-1, g.shape[-1])  # [N*S, O]
            local_update_oo = g_bo.mT @ g_bo
            dist.all_reduce(local_update_oo, op=dist.ReduceOp.SUM)

            start_row = self.rank * S_cov_po.shape[0]
            end_row = (self.rank + 1) * S_cov_po.shape[0]
            update_slice_po = local_update_oo[start_row:end_row, :]
            # Add it to permanently stored slice
            S_cov_po.add_(update_slice_po)

        collector = EkfacCollector(
            self.model.base_model,
            closure=callback_gradient,
            processor=self.processor,
            target_modules=self.target_modules,
            fwd_closure=callback_activation,
        )

        # main computation takes place here
        total_processed = self._collector(collector, desc="covariances")

        if dist.is_initialized():
            dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)

        if self.rank == 0:
            torch.save(total_processed, os.path.join(self.path, "total_processed.pt"))
            print(f"Total processed: {total_processed.item()}")

        activation_path = os.path.join(self.path, "activation_covariance_sharded")
        gradient_path = os.path.join(self.path, "gradient_covariance_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)

        # Save the sharded covariance matrices
        save_file(A_cov_dict, os.path.join(activation_path, f"shard_{self.rank}.safetensors"))
        save_file(S_cov_dict, os.path.join(gradient_path, f"shard_{self.rank}.safetensors"))

        if self.rank == 0:
            print(f"Saved activation covariance to {activation_path}")
            print(f"Saved gradient covariance to {gradient_path}")
            print("-*-" * 50)
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()

    def compute_eigendecomposition(self, covariance_type: Literal["activation", "gradient"]):
        """Computes the eigendecomposition of the covariance matrices."""
        total_processed = torch.load(
            os.path.join(self.path, "total_processed.pt"),
            map_location=f"cuda:{self.rank}",
        )

        random.seed(0)
        shuffled_target_info = random.sample(list(self.target_info), len(list(self.target_info)))

        target_info_rank = shuffled_target_info[self.rank :: self.world_size]

        covariance_eigenvectors = {}

        for key in tqdm(
            target_info_rank,
            disable=False,
            desc=f"Rank {self.rank}: Computing {covariance_type} eigenvectors",
            position=self.rank,
            leave=False,
        ):
            matrix = self._compute_full_matrix(key, covariance_type=covariance_type)
            original_dtype = matrix.dtype
            matrix_normalized = matrix.to(torch.float64) / total_processed
            matrix_normalized = (matrix_normalized + matrix_normalized.T).div(2)

            # check if matrix_normalized is has NaNs or Infs
            if not torch.isfinite(matrix_normalized).all():
                raise ValueError(
                    f"Covariance matrix for {key} of type {covariance_type} contains NaNs or Infs."
                    "Consider increasing to fp32."
                )
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(matrix_normalized)

            except Exception as e:
                raise RuntimeError(f"Eigendecomposition failed for {key} of type {covariance_type}") from e

            eigenvectors = eigenvectors.to(original_dtype).to(device="cpu").contiguous()
            covariance_eigenvectors[key] = eigenvectors
            if self.rank == 0 and self.debug:
                print(key, eigenvectors.sum())

        covariance_eigenvectors = self._merge_and_shard_dict(
            input_dict=covariance_eigenvectors, covariance_type=covariance_type
        )

        eigen_path = os.path.join(self.path, f"{covariance_type}_eigen_sharded")

        os.makedirs(eigen_path, exist_ok=True)

        save_file(
            covariance_eigenvectors,
            os.path.join(eigen_path, f"shard_{self.rank}.safetensors"),
        )
        if self.rank == 0:
            print(f"Saved {covariance_type} eigenvectors to {eigen_path}")
        gc.collect()
        torch.cuda.empty_cache()

    def compute_eigenvalue_correction(
        self,
    ):
        """
        Compute projected gradients using a subset of the dataset.
        """

        if self.rank == 0:
            print(f"Computing eigenvalue correction for {len(self.data)} documents...")

        eigen_a = load_file(
            self.path + f"/activation_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        eigen_g = load_file(
            self.path + f"/gradient_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )

        eigenvalue_corrections = {}
        transformed_activation_cache = {}

        # def callback_activation(name: str, a: torch.Tensor):
        #     a = a.reshape(-1, a.shape[-1])  # [N*S, O]

        #     a_chunks = torch.chunk(a, self.world_size, dim=1)

        #     output_dim = eigen_a[name].shape[1]
        #     result = torch.zeros(a.shape[0], output_dim, device=a.device, dtype=a.dtype)

        #     for rank_index in range(self.world_size):
        #         if rank_index == self.rank:
        #             A_shard = eigen_a[name]
        #         else:
        #             A_shard = torch.zeros_like(eigen_a[name])

        #         dist.broadcast(A_shard, src=rank_index)
        #         # Accumulate partial result
        #         result += a_chunks[rank_index] @ A_shard  # [N*S, D]

        #         # Clean up
        #         if self.rank != rank_index:
        #             del A_shard
        #     transformed_activation_cache[name] = result

        #     if self.rank == 0 and self.debug:
        #         run_covariances_shards = [
        #             os.path.join(
        #                 self.path,
        #                 "activation_eigen_sharded",
        #                 f"shard_{rank}.safetensors",
        #             )
        #             for rank in range(self.world_size)
        #         ]
        #         run_covariances_list = [(load_file(shard)) for shard in run_covariances_shards]
        #         run_covariances = {}
        #         for k, v in run_covariances_list[0].items():
        #             run_covariances[k] = torch.cat([shard[k] for shard in run_covariances_list], dim=0).to(a.device)

        #         result_2 = a @ run_covariances[name]
        #         assert torch.allclose(result, result_2, atol=1e-4, rtol=1e-4), (
        #             "Distributed eigenvector multiplication failed"
        #         )

        # def callback_gradient(name: str, g: torch.Tensor):
        #     g = g.reshape(-1, g.shape[-1])  # [N*S, I]

        #     g_chunks = torch.chunk(g, self.world_size, dim=1)
        #     output_dim = eigen_g[name].shape[1]

        #     result = torch.zeros(g.shape[0], output_dim, device=g.device, dtype=g.dtype)

        #     for rank_index in range(self.world_size):
        #         if rank_index == self.rank:
        #             G_shard = eigen_g[name]

        #         else:
        #             G_shard = torch.zeros_like(eigen_g[name])

        #         dist.broadcast(G_shard, src=rank_index)

        #         # Accumulate partial result

        #         result += torch.einsum(" r l, b r-> b l", G_shard, g_chunks[rank_index])

        #         # Clean up
        #         if rank_index != self.rank:
        #             del G_shard

        #     transformed_grad_shard = torch.einsum(
        #         "B O, B K-> K O", transformed_activation_cache[name] ** 2, result**2
        #     ).contiguous()
        #     dist.all_reduce(transformed_grad_shard, op=dist.ReduceOp.SUM)

        #     start_row = self.rank * transformed_grad_shard.shape[0]
        #     end_row = (self.rank + 1) * transformed_grad_shard.shape[0]
        #     if name not in eigenvalue_corrections:
        #         eigenvalue_corrections[name] = transformed_grad_shard[start_row:end_row, :].to(
        #             device="cpu", non_blocking=True
        #         )
        #     else:
        #         eigenvalue_corrections[name].to(device=self.device).add_(
        #             transformed_grad_shard[start_row:end_row, :]
        #         ).to(device="cpu", non_blocking=True)

        #     del transformed_grad_shard

        #     if self.rank == 0 and self.debug:
        #         run_covariances_shards = [
        #             os.path.join(self.path, "gradient_eigen_sharded", f"shard_{rank}.safetensors")
        #             for rank in range(self.world_size)
        #         ]
        #         run_covariances_list = [(load_file(shard)) for shard in run_covariances_shards]

        #         run_covariances = torch.cat([shard[name] for shard in run_covariances_list], dim=0).to(g.device)
        #         result_2 = torch.einsum(" r l, b r-> b l", run_covariances, g)

        #         assert torch.allclose(result, result_2, atol=1e-0, rtol=1e-4), (
        #             "Distributed eigenvector multiplication failed"
        #         )

        def callback_activation(name: str, a: torch.Tensor):
            a = a.reshape(-1, a.shape[-1])  # [N*S, O]

            a_chunks = torch.chunk(a, self.world_size, dim=1)

            output_dim = eigen_a[name].shape[1]
            result = torch.zeros(a.shape[0], output_dim, device=a.device, dtype=a.dtype)

            for rank_index in range(self.world_size):
                if rank_index == self.rank:
                    A_shard = eigen_a[name]
                else:
                    A_shard = torch.zeros_like(eigen_a[name]).contiguous()

                dist.broadcast(A_shard, src=rank_index)
                # Accumulate partial result
                result += a_chunks[rank_index] @ A_shard  # [N*S, D]

                # Clean up
                if self.rank != rank_index:
                    del A_shard
            transformed_activation_cache[name] = result

            if self.rank == 0:
                run_covariances_shards = [
                    os.path.join(
                        self.path,
                        "activation_eigen_sharded",
                        f"shard_{rank}.safetensors",
                    )
                    for rank in range(self.world_size)
                ]
                run_covariances_list = [(load_file(shard)) for shard in run_covariances_shards]
                run_covariances = {}
                for k, v in run_covariances_list[0].items():
                    run_covariances[k] = torch.cat([shard[k] for shard in run_covariances_list], dim=0).to(a.device)

                result_2 = a @ run_covariances[name]
                assert torch.allclose(result, result_2, atol=1e-4, rtol=1e-4), (
                    "Distributed eigenvector multiplication failed"
                )

            # if self.rank == 0 and name == "layers.0.mlp.dense_4h_to_h" and self.debug:
            #     print(f"{self.rank}: {result.abs().sum()}")

        def callback_gradient(name: str, g: torch.Tensor):
            g = g.reshape(-1, g.shape[-1])  # [N*S, I]

            g_chunks = torch.chunk(g, self.world_size, dim=1)
            output_dim = eigen_g[name].shape[1]

            result = torch.zeros(g.shape[0], output_dim, device=g.device, dtype=g.dtype)

            for rank_index in range(self.world_size):
                if rank_index == self.rank:
                    G_shard = eigen_g[name]

                else:
                    G_shard = torch.zeros_like(eigen_g[name]).contiguous()

                dist.broadcast(G_shard, src=rank_index)

                # Accumulate partial result

                result += torch.einsum(" r l, b r-> b l", G_shard, g_chunks[rank_index])

                # Clean up
                if rank_index != self.rank:
                    del G_shard

            transformed_grad_shard = torch.einsum(
                "B O, B K-> K O", transformed_activation_cache[name] ** 2, result**2
            ).contiguous()
            dist.all_reduce(transformed_grad_shard, op=dist.ReduceOp.SUM)

            start_row = self.rank * transformed_grad_shard.shape[0]
            end_row = (self.rank + 1) * transformed_grad_shard.shape[0]
            if name not in eigenvalue_corrections:
                eigenvalue_corrections[name] = transformed_grad_shard[start_row:end_row, :].contiguous()
            else:
                eigenvalue_corrections[name].add_(transformed_grad_shard[start_row:end_row, :].contiguous())

            if self.rank == 0:
                run_covariances_shards = [
                    os.path.join(self.path, "gradient_eigen_sharded", f"shard_{rank}.safetensors")
                    for rank in range(self.world_size)
                ]
                run_covariances_list = [(load_file(shard)) for shard in run_covariances_shards]

                run_covariances = torch.cat([shard[name] for shard in run_covariances_list], dim=0).to(g.device)
                result_2 = torch.einsum(" r l, b r-> b l", run_covariances, g)

                assert torch.allclose(result, result_2, atol=1e-0, rtol=1e-4), (
                    "Distributed eigenvector multiplication failed"
                )

        collector = EkfacCollector(
            self.model.base_model,
            closure=callback_gradient,
            processor=self.processor,
            target_modules=self.target_modules,
            fwd_closure=callback_activation,
        )

        total_processed = self._collector(collector, desc="eigenvalue correction")

        if dist.is_initialized():
            dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            torch.save(total_processed, os.path.join(self.path, "total_processed_lambda.pt"))
            print(f"Total processed: {total_processed.item()}")
        for k, v in eigenvalue_corrections.items():
            v.div_(total_processed.to(device=v.device))

        os.makedirs(self.path + "/eigenvalue_correction_sharded", exist_ok=True)
        save_file(
            eigenvalue_corrections,
            self.path + f"/eigenvalue_correction_sharded/shard_{self.rank}.safetensors",
        )

    @staticmethod
    def _sharded_matmul(vector: Float[Tensor, "B a"], matrix: Float[Tensor, "a_i b"], world_size: int):
        """
        Sharded matrix multiplication for distributed training.
        vector: [B, a]
        matrix_shard: [a_i, b]
        Returns: [B, b]
        """
        # Split the vector into shards
        vector_shards = torch.chunk(vector, dist.get_world_size(), dim=1)

        result = torch.zeros(vector.shape[0], matrix.shape[1], device=vector.device, dtype=vector.dtype)

        for rank_index in range(dist.get_world_size()):
            if rank_index == dist.get_rank():
                shard = vector_shards[rank_index]
            else:
                shard = torch.zeros_like(vector_shards[rank_index])

            dist.broadcast(shard, src=rank_index)
            result += shard @ matrix

        return result

    def _setup_profiler(self):
        """Set up profiler if profiling is enabled."""
        if not self.profile:
            return nullcontext()

        trace_handler = tensorboard_trace_handler(
            dir_name="profiler_logs", worker_name=f"rank_{self.rank}", use_gzip=True
        )
        my_schedule = schedule(wait=0, warmup=0, active=4, repeat=1)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=trace_handler,
            schedule=my_schedule,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_modules=True,
        )

        log_dir = "profiler_logs"
        os.makedirs(log_dir, exist_ok=True)

        return prof

    def _sharded_covariance(
        self,
        activation_covariance_dict: dict,
        gradient_covariance_dict: dict,
        dtype: torch.dtype,
    ):
        """This function initializes the sharded covariance matrices for activations and gradients.
        So far we always shard using the first dimension.
        TODO: Make this also work when dimension is not divisible by world_size."""

        for name, (device, weight_shape) in self.target_info.items():
            # Activation covariance A^T A has shape [in_dim, in_dim]
            in_dim = weight_shape[1]
            if in_dim % self.world_size != 0:
                raise ValueError(f"Activation dim {in_dim} for {name} not divisible by world_size {self.world_size}")
            shard_size = in_dim // self.world_size
            activation_covariance_dict[name] = torch.zeros((shard_size, in_dim), device=self.device, dtype=dtype)

            # Gradient covariance G^T G has shape [out_dim, out_dim]
            out_dim = weight_shape[0]
            if out_dim % self.world_size != 0:
                raise ValueError(f"Gradient dim {out_dim} for {name} not divisible by world_size {self.world_size}")
            shard_size = out_dim // self.world_size
            gradient_covariance_dict[name] = torch.zeros((shard_size, out_dim), device=self.device, dtype=dtype)

    def _collector(self, collector, desc: Optional[str] = None) -> Float[Tensor, " "]:
        total_processed = torch.tensor(0, device=self.model.device)
        prof = self._setup_profiler()
        step = 0
        try:
            # torch.cuda.memory._record_memory_history()
            with prof:
                for sl in tqdm(self.batches, disable=self.rank != 0, desc=f"Computing {desc}"):
                    batch = self.data[sl]
                    x, y = pad_and_tensor(
                        batch["input_ids"],  # type: ignore
                        labels=batch.get("labels"),  # type: ignore
                        device=self.model.device,
                    )

                    total_processed += x.numel()

                    with record_function(f"step_{step}") if self.profile else nullcontext():
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

                    if self.profile:
                        assert isinstance(prof, profile), "Profiler is not set up correctly"
                        prof.step()
                    step += 1

        finally:
            pass
            # torch.cuda.memory._dump_snapshot(f"snapshot_{self.rank}.pickle")

        return total_processed

    def _compute_full_matrix(self, name: str, covariance_type: Literal["activation", "gradient"]):
        """
        Load a full matrix from sharded covariance files. Needed to compute eigendecomposition.
        """
        shard_path = os.path.join(self.path, f"{covariance_type}_covariance_sharded")
        files = os.listdir(shard_path)
        assert len(files) == self.world_size, f"Expected {self.world_size} shards, found {len(files)} in {self.path}"

        full_matrix = []
        for shard_id in range(self.world_size):
            shard_path_rank = os.path.join(shard_path, f"shard_{shard_id}.safetensors")
            with safe_open(shard_path_rank, framework="pt", device=f"cuda:{self.rank}") as f:
                local_matrix = f.get_tensor(name)

            full_matrix.append(local_matrix)

        # Concatenate all shards to form the full matrix
        full_matrix = torch.cat(full_matrix, dim=0)

        assert full_matrix.shape[0] == full_matrix.shape[1], "Full covariance matrix must be square"

        return full_matrix

    def _merge_and_shard_dict(
        self,
        input_dict: dict[str, torch.Tensor],
        covariance_type: Literal["activation", "gradient"],
    ):
        """This function takes a dict of tensors, where each rank will have eigenvectors of *some* modules.
        It then redistributes the tensors across all ranks,
        so that each rank has a shard of the eigenvector of *each* module.
        """

        for key in self.target_info:
            shard_size = self.target_info[key][1]
            d_in, d_out = self.target_info[key][1]
            d = d_out if covariance_type == "activation" else d_in
            shard_size = d // self.world_size

            if key not in input_dict:
                tensor = torch.zeros([d, d], device=self.device, dtype=self.dtype)
            else:
                tensor = input_dict[key].to(device=self.device)

            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            shard = torch.empty(shard_size, d, device=self.device, dtype=self.dtype)
            shard.copy_(tensor[self.rank * shard_size : (self.rank + 1) * shard_size, :])
            input_dict[key] = shard.to(device="cpu", non_blocking=True)

            assert shard.shape[0] == shard_size, f"Shard shape {shard.shape} does not match expected {shard_size}"

            del tensor

            gc.collect()
            torch.cuda.empty_cache()

        return input_dict

    def compute_ivhp(self):
        def callback_activation(name: str, a: Float[Tensor, "N S I"], projection_act: Float[Tensor, "I P"]):
            """Forward hook to compute the covariance of activations A_l."""

            # TODO: Sharded MVP

            a = a @ projection_act

        def callback_gradient(name: str, g: Float[Tensor, "N S O"], projection_grad: Float[Tensor, "O Q"]):
            """Backward hook to compute the covariance of pseudo-gradients S_{l-1}."""

            # TODO: Sharded MVP

            g = g @ projection_grad

        collector = EkfacCollector(
            self.model.base_model,
            closure=callback_gradient,
            processor=self.processor,
            target_modules=self.target_modules,
            fwd_closure=callback_activation,
        )

        # main computation takes place here
        total_processed = self._collector(collector, desc="covariances")

        if dist.is_initialized():
            dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)

        if self.rank == 0:
            torch.save(total_processed, os.path.join(self.path, "total_processed.pt"))
            print(f"Total processed: {total_processed.item()}")

        activation_path = os.path.join(self.path, "activation_covariance_sharded")
        gradient_path = os.path.join(self.path, "gradient_covariance_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)

        if self.rank == 0:
            print(f"Saved activation covariance to {activation_path}")
            print(f"Saved gradient covariance to {gradient_path}")
            print("-*-" * 50)
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
