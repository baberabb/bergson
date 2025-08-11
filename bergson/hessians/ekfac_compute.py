import gc
import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from typing import Literal, Optional

import numpy as np
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

from bergson.data import IndexConfig, create_index, load_gradients, pad_and_tensor
from bergson.gradients import (
    GradientProcessor,
)
from bergson.hessians.collector import EkfacCollector
from bergson.hessians.logger import get_logger


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
        *,
        batches: list[list[int]],
        target_modules: set[str] | None = None,
        cfg: IndexConfig,
    ):
        self.model = model

        self.device = model.device
        self.dtype = model.dtype
        self.ekfac_collector = EkfacCollector(model.base_model, target_modules=target_modules)
        self.target_info = self.ekfac_collector.target_info

        self.processor = processor
        self.target_modules = target_modules  # which modules to compute EKFAC for, by default uses all MLPs
        self.data = data
        self.batches = batches
        self.path = os.path.join(cfg.run_path, "influence_results")

        self.cfg = cfg

        self.logger = get_logger("EkfacComputer", level="DEBUG" if cfg.debug else "INFO")

        ### Distributed related
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        if self.rank == 0:
            self.logger.info(f"Computing EKFAC for {list(self.target_info)} target modules.")

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
            self.logger.info(f"Total processed: {total_processed.item()}")

        activation_path = os.path.join(self.path, "activation_covariance_sharded")
        gradient_path = os.path.join(self.path, "gradient_covariance_sharded")

        os.makedirs(activation_path, exist_ok=True)
        os.makedirs(gradient_path, exist_ok=True)

        # Save the sharded covariance matrices
        save_file(A_cov_dict, os.path.join(activation_path, f"shard_{self.rank}.safetensors"))
        save_file(S_cov_dict, os.path.join(gradient_path, f"shard_{self.rank}.safetensors"))

        if self.rank == 0:
            self.logger.info(f"Saved activation covariance to {activation_path}")
            self.logger.info(f"Saved gradient covariance to {gradient_path}")
            self.logger.info("-*-" * 50)
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()

    def compute_eigendecomposition(self, covariance_type: Literal["activation", "gradient"]):
        """This is Eq. 18 from above reference."""
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
            matrix = self._compute_full_matrix(key, shard_type=covariance_type + "_covariance")  # type: ignore

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
            self.logger.info(f"Saved {covariance_type} eigenvectors to {eigen_path}")
        gc.collect()
        torch.cuda.empty_cache()

    def compute_eigenvalue_correction(
        self,
    ):
        """
        This is Eq. 20 from above reference.
        """

        if self.rank == 0:
            self.logger.info(f"Computing eigenvalue correction for {len(self.data)} documents...")

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

        def callback_activation(name: str, a: torch.Tensor):
            a_ni = a.reshape(-1, a.shape[-1])  # [N*S, I]

            transformed_activation_cache[name] = self._sharded_vec_matmul(
                vector_na=a_ni, matrix_cb=eigen_a[name], mult_type="left"
            )  # shape [N*S, I]

            if self.rank == 0 and self.cfg.debug:
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

                result_2 = a_ni @ run_covariances[name]
                assert torch.allclose(transformed_activation_cache[name], result_2, atol=1e-4, rtol=1e-4), (
                    "Distributed eigenvector multiplication failed"
                )

        def callback_gradient(name: str, g: torch.Tensor):
            g_no = g.reshape(-1, g.shape[-1])  # [N*S, O]

            result_no = self._sharded_vec_matmul(vector_na=g_no, matrix_cb=eigen_g[name], mult_type="right")  # [N*S, O]

            transformed_grad_shard = torch.einsum(
                "B I, B O-> O I", transformed_activation_cache[name] ** 2, result_no**2
            ).contiguous()  # [O, I]
            dist.all_reduce(transformed_grad_shard, op=dist.ReduceOp.SUM)

            shard_size = transformed_grad_shard.shape[0] // self.world_size
            start_row = self.rank * shard_size
            end_row = (self.rank + 1) * shard_size
            if name not in eigenvalue_corrections:
                eigenvalue_corrections[name] = (
                    transformed_grad_shard[start_row:end_row, :].contiguous().to(device="cpu", non_blocking=False)
                )
            else:
                eigenvalue_corrections[name] = eigenvalue_corrections[name].to(device=self.device)
                eigenvalue_corrections[name].add_(transformed_grad_shard[start_row:end_row, :].contiguous())
                eigenvalue_corrections[name] = eigenvalue_corrections[name].to(device="cpu", non_blocking=False)

            if self.rank == 0 and self.cfg.debug:
                run_covariances_shards = [
                    os.path.join(self.path, "gradient_eigen_sharded", f"shard_{rank}.safetensors")
                    for rank in range(self.world_size)
                ]
                run_covariances_list = [(load_file(shard)) for shard in run_covariances_shards]

                run_covariances = torch.cat([shard[name] for shard in run_covariances_list], dim=0).to(g.device)
                result_2 = torch.einsum(" r l, b r-> b l", run_covariances, g_no)

                assert torch.allclose(result_no, result_2, atol=1e-0, rtol=1e-4), (
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
            self.logger.info(f"Total processed: {total_processed.item()}")
        for k, v in eigenvalue_corrections.items():
            v.div_(total_processed.to(device=v.device))

        os.makedirs(self.path + "/eigenvalue_correction_sharded", exist_ok=True)
        save_file(
            eigenvalue_corrections,
            self.path + f"/eigenvalue_correction_sharded/shard_{self.rank}.safetensors",
        )

    def _sharded_vec_matmul(
        self,
        vector_na: Float[Tensor, "n a"],
        matrix_cb: Float[Tensor, "c b"],
        mult_type: Literal["left", "right"],
    ):
        """
        Sharded matrix multiplication for distributed training. Assumes that c= a/world_size.
        vector: [n, a]
        matrix_shard: [c, b]
        Returns: [n, b]
        """
        # Split the vector into shards
        vector_shards_wnc = torch.chunk(vector_na, self.world_size, dim=1)  # (w, n, c)

        result_nb = torch.zeros(
            vector_na.shape[0],
            matrix_cb.shape[1],
            device=vector_na.device,
            dtype=vector_na.dtype,
        )

        for rank_index in range(self.world_size):
            if rank_index == self.rank:
                shard_cb = matrix_cb
            else:
                shard_cb = torch.zeros_like(matrix_cb)

            dist.broadcast(shard_cb, src=rank_index)
            if mult_type == "left":
                result_nb += torch.einsum("n c, c b-> n b", vector_shards_wnc[rank_index], shard_cb)  # [B, c]
            elif mult_type == "right":
                result_nb += torch.einsum("c b, n c-> n b", shard_cb, vector_shards_wnc[rank_index])
            if self.rank != rank_index:
                del shard_cb

        return result_nb

    def _setup_profiler(self):
        """Set up profiler if profiling is enabled."""
        if not self.cfg.profile:
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

                    with record_function(f"step_{step}") if self.cfg.profile else nullcontext():
                        with collector:
                            logits = self.model(x).logits
                            logits = logits[:, :-1].reshape(-1, logits.size(-1))

                            if not self.cfg.sample:
                                losses = F.cross_entropy(
                                    logits,
                                    y[:, 1:].flatten(),
                                    reduction="none",
                                ).reshape_as(y[:, 1:])
                            else:
                                with torch.no_grad():
                                    probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                                    sampled_labels = torch.multinomial(
                                        probs,
                                        num_samples=1,
                                    ).flatten()

                                    del probs

                                losses = F.cross_entropy(
                                    logits,
                                    sampled_labels,
                                    reduction="none",
                                ).reshape_as(y[:, 1:])

                                masks = y[:, 1:] != -100
                                denoms = masks.sum(dim=1, dtype=logits.dtype)
                                losses = losses.sum(1).div(denoms)

                            losses.mean().backward()
                            self.model.zero_grad()

                    if self.cfg.profile:
                        assert isinstance(prof, profile), "Profiler is not set up correctly"
                        prof.step()
                    step += 1

        finally:
            # torch.cuda.memory._dump_snapshot(f"snapshot_{self.rank}.pickle")
            pass

        return total_processed

    def _compute_full_matrix(
        self,
        name: str,
        shard_type: Literal[
            "activation_covariance",
            "gradient_covariance",
            "eigenvalue_correction",
            "inverse_eigenvalue_correction",
        ],
    ):
        """
        Load a full matrix from sharded covariance files. Needed to compute eigendecomposition.
        """
        shard_path = os.path.join(self.path, f"{shard_type}_sharded")
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

        return full_matrix

    def _merge_and_shard_dict(
        self,
        input_dict: dict[str, torch.Tensor],
        covariance_type: Literal["activation", "gradient"],
    ):
        """This function takes a dict of tensors, where each rank will have eigenvectors of *some* modules.
        It then redistributes the tensors across all ranks,
        so that each rank has a shard of the eigenvectors of *each* module.
        """

        for key in self.target_info:
            d_out, d_in = self.target_info[key][1]
            d = d_in if covariance_type == "activation" else d_out
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


class EkfacApplicator:
    def __init__(
        self,
        cfg: IndexConfig,
    ):
        self.cfg = cfg
        self.path = os.path.join(cfg.run_path, "influence_results")
        self.gradient_path = cfg.gradient_path
        # hash of self.path
        path_hash = hashlib.md5(self.path.encode("utf-8")).hexdigest()
        self.gradient_ekfac_path = self.gradient_path + path_hash
        os.makedirs(self.gradient_ekfac_path, exist_ok=True)
        self.logger = get_logger("EkfacApplicator", level="DEBUG" if cfg.debug else "INFO")

        ### FSDP related
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = f"cuda:{self.rank}"

        match cfg.precision:
            case "bf16":
                self.dtype = torch.bfloat16
            case "fp16":
                self.dtype = torch.float16
            case "fp32":
                self.dtype = torch.float32
            case "int4" | "int8":
                self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            case other:
                raise ValueError(f"Unsupported precision: {other}")

    def prepare_attribution(self):
        eigen_a = load_file(
            self.path + f"/activation_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        eigen_g = load_file(
            self.path + f"/gradient_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )

        random_eigen_a = {}
        random_eigen_g = {}

        p = self.cfg.projection_dim

        for name in eigen_a.keys():
            proj_pi = self._projection(
                name,
                p,  # type: ignore
                eigen_a[name].shape[1],
                side="right",
                dtype=eigen_a[name].dtype,
            )

            proj_shards_wpt = torch.chunk(proj_pi, self.world_size, dim=1)  # (w, p, i/w)
            result_shard_pi = torch.einsum("t i, p t-> p i", eigen_a[name], proj_shards_wpt[self.rank]).contiguous()

            dist.all_reduce(result_shard_pi, op=dist.ReduceOp.SUM)

            shard_size = result_shard_pi.shape[0] // self.world_size
            start_row = self.rank * shard_size
            end_row = (self.rank + 1) * shard_size

            random_eigen_a[name] = result_shard_pi[start_row:end_row, :]

        random_activation_path = os.path.join(self.path, "random_activation_eigen_sharded")
        os.makedirs(random_activation_path, exist_ok=True)
        save_file(
            random_eigen_a,
            os.path.join(random_activation_path, f"shard_{self.rank}.safetensors"),
        )

        for name in eigen_g.keys():
            proj_qo = self._projection(
                name,
                p,  # type: ignore
                eigen_g[name].shape[1],
                side="left",
                dtype=eigen_g[name].dtype,
            )
            proj_shards_wqr = torch.chunk(proj_qo, self.world_size, dim=1)  # (w, q, o/w)
            result_shard_qo = torch.einsum("q r, r o -> q o", proj_shards_wqr[self.rank], eigen_g[name]).contiguous()
            dist.all_reduce(result_shard_qo, op=dist.ReduceOp.SUM)

            shard_size = result_shard_qo.shape[0] // self.world_size
            start_row = self.rank * shard_size
            end_row = (self.rank + 1) * shard_size
            random_eigen_g[name] = result_shard_qo[start_row:end_row, :]

        random_gradient_path = os.path.join(self.path, "random_gradient_eigen_sharded")
        os.makedirs(random_gradient_path, exist_ok=True)
        save_file(
            random_eigen_g,
            os.path.join(random_gradient_path, f"shard_{self.rank}.safetensors"),
        )

        if self.rank == 0:
            self.logger.info(f"Saved random activation eigenvectors to {random_activation_path}")
            self.logger.info(f"Saved random gradient eigenvectors to {random_gradient_path}")

            self.logger.info("-*-" * 50)

    def compute_ivhp_sharded(self):
        eigen_a = load_file(
            self.path + f"/activation_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        eigen_g = load_file(
            self.path + f"/gradient_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )

        random_eigen_a = load_file(
            self.path + f"/random_activation_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )
        random_eigen_g = load_file(
            self.path + f"/random_gradient_eigen_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )

        lambda_factor = load_file(
            self.path + f"/eigenvalue_correction_sharded/shard_{self.rank}.safetensors",
            device=f"cuda:{self.rank}",
        )

        for k, v in lambda_factor.items():
            eigen_a[k] = eigen_a[k].to(dtype=torch.float32)
            eigen_g[k] = eigen_g[k].to(dtype=torch.float32)
            random_eigen_a[k] = random_eigen_a[k].to(dtype=torch.float32)
            random_eigen_g[k] = random_eigen_g[k].to(dtype=torch.float32)
            lambda_factor[k] = v.to(dtype=torch.float32)

        grad_sizes = {
            name: random_eigen_g[name].shape[0] * self.world_size * random_eigen_a[name].shape[0] * self.world_size
            for name in random_eigen_a
        }

        mmap = load_gradients(self.gradient_path)
        with open(os.path.join(self.gradient_path, "info.json")) as f:
            info = json.load(f)

        # Allocate structured space ahead of time for the gradients
        grad_buffer = create_index(
            self.gradient_ekfac_path,
            num_grads=info["num_grads"],
            grad_sizes=grad_sizes,
            dtype=np.float32,
        )

        if self.rank == 0:
            self.logger.info(f"Loaded gradients for {len(mmap)} queries and computing IVHP...")

        for i in tqdm(
            range(math.ceil(info["num_grads"] / self.cfg.gradient_batch_size)),
            desc="IVHP batches",
            disable=self.rank != 0,
        ):
            batch_slice = slice(
                i * self.cfg.gradient_batch_size, min((i + 1) * self.cfg.gradient_batch_size, info["num_grads"])
            )

            transformed_gradients_slice = self.compute_ivhp_batch(
                eigen_a=eigen_a,
                mmap=mmap,
                eigen_g=eigen_g,
                lambda_factor=lambda_factor,
                random_eigen_a=random_eigen_a,
                random_eigen_g=random_eigen_g,
                batch_slice=batch_slice,
            )
            torch.cuda.synchronize()
            for k, v in transformed_gradients_slice.items():
                v = v.to(device="cpu", non_blocking=True).flatten(1).numpy()
                grad_buffer[k][batch_slice] = v
            transformed_gradients_slice.clear()

        grad_buffer.flush()
        if self.rank == 0:
            self.logger.info(f"Saved IVHP gradients to {self.gradient_ekfac_path}")

    def compute_ivhp_batch(self, eigen_a, mmap, eigen_g, lambda_factor, random_eigen_a, random_eigen_g, batch_slice):
        transformed_gradients = {}
        for k, v in eigen_a.items():
            gradients_noi = torch.from_numpy(mmap[k][batch_slice].copy()).to(
                device=self.device, dtype=torch.float32
            )  # shape [num_grads, out*in]
            gradients_noi = gradients_noi.view(-1, eigen_g[k].shape[1], eigen_a[k].shape[1])
            transformed_gradients[k] = self._sharded_matmul(gradients_noi=gradients_noi, matrix_cb=v, mult_type="left")

        if self.rank == 0:
            self.logger.debug("Finished G @ Q_A")

        del eigen_a
        gc.collect()
        torch.cuda.empty_cache()

        for k, v in eigen_g.items():
            transformed_gradients[k] = self._sharded_matmul(
                gradients_noi=transformed_gradients[k], matrix_cb=v, mult_type="right", k=k
            )

        if self.rank == 0:
            self.logger.debug("Finished G'=Q_S.T @ G @ Q_A")
        del eigen_g
        gc.collect()
        torch.cuda.empty_cache()

        for k, v in lambda_factor.items():
            self._sharded_mult(matrix_noi=transformed_gradients[k], lambda_ci=v, k=k)  # this is in-place

        if self.rank == 0:
            self.logger.debug("Finished G'/lambda")

        for k, v in random_eigen_a.items():
            transformed_gradients[k] = self._sharded_transpose_matmul(
                matrix_noi=transformed_gradients[k], matrix_bc=v, mult_type="left"
            )
        if self.rank == 0:
            self.logger.debug("Finished G'/lambda @ P_A.T")

        for k, v in random_eigen_g.items():
            transformed_gradients[k] = self._sharded_transpose_matmul(
                matrix_noi=transformed_gradients[k], matrix_bc=v, mult_type="right"
            )

        if self.rank == 0:
            self.logger.debug("Finished P_S.T @ G'/lambda @ P_A.T")
        return transformed_gradients

    def _projection(
        self,
        name: str,
        m: int,
        n: int,
        side: Literal["left", "right"],
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the `side` projection matrix for parameter `name` of shape [m, n]."""
        # Seed the PRNG with the name of the layer and what "side" we are projecting
        message = bytes(f"{name}/{side}", "utf-8")
        digest = hashlib.md5(message).digest()
        seed = int.from_bytes(digest, byteorder="big") % (2**63 - 1)
        device = self.device
        prng = torch.Generator(device).manual_seed(seed)

        A = torch.randn(m, n, device=device, dtype=dtype, generator=prng)
        A /= A.norm(dim=1, keepdim=True)
        return A

    def _compute_full_matrix(
        self,
        name: str,
        shard_type: Literal[
            "activation_covariance",
            "gradient_covariance",
            "eigenvalue_correction",
            "inverse_eigenvalue_correction",
        ],
    ):
        """
        Load a full matrix from sharded covariance files. Needed to compute eigendecomposition.
        """
        shard_path = os.path.join(self.path, f"{shard_type}_sharded")
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

        return full_matrix

    def _sharded_matmul(
        self,
        gradients_noi: Float[Tensor, "n o i"],
        matrix_cb: Float[Tensor, "c b"],
        mult_type: Literal["left", "right"],
        k=None,
    ):
        """
        Sharded matrix multiplication for distributed training.
        Assumes that c=i/world_size if left or o/world_size if right.
        gradients: [n, o, i]
        matrix_shard: [c, b] where c=i/w if left or c=o/w if right
        Returns: [n, o, b] if left or [n, b, i] if right
        """
        # Split the vector into shards
        gradients_shard_wnfg = torch.chunk(
            gradients_noi, self.world_size, dim=2 if mult_type == "left" else 1
        )  # (w, n, f, g) where f=i/w if left or g=o/w if right

        x, y = (
            (gradients_noi.shape[1], matrix_cb.shape[1])
            if mult_type == "left"
            else (matrix_cb.shape[1], gradients_noi.shape[2])
        )
        result_nxy = torch.zeros(
            gradients_noi.shape[0],
            x,
            y,
            device=gradients_noi.device,
            dtype=gradients_noi.dtype,
        )

        for rank_index in range(self.world_size):
            if rank_index == self.rank:
                shard_cb = matrix_cb
            else:
                shard_cb = torch.zeros_like(matrix_cb)

            dist.broadcast(shard_cb, src=rank_index)
            if mult_type == "left":
                result_nxy += torch.einsum("n o c, c b-> n o b", gradients_shard_wnfg[rank_index], shard_cb)  # [B, c]
            elif mult_type == "right":
                result_nxy += torch.einsum("c b, n c i -> n b i", shard_cb, gradients_shard_wnfg[rank_index])
            if self.rank != rank_index:
                del shard_cb

        return result_nxy

    def _sharded_mult(
        self, matrix_noi: Float[Tensor, "n o i"], lambda_ci: Float[Tensor, "c i"], k: Optional[str] = None
    ):
        """
        Sharded in-place element-wise division for distributed training.
        gradients: [n, o, i]
        matrix_shard: [c, i] where c=o/world_size

        """

        global_lambda_mean = lambda_ci.mean()

        dist.all_reduce(global_lambda_mean, op=dist.ReduceOp.SUM)
        global_lambda_mean /= self.world_size

        for rank_index in range(self.world_size):
            if rank_index == self.rank:
                shard_ci = lambda_ci
            else:
                shard_ci = torch.zeros_like(lambda_ci)

            dist.broadcast(shard_ci, src=rank_index)

            start_row = rank_index * shard_ci.shape[0]
            end_row = (rank_index + 1) * shard_ci.shape[0]
            inverse_lambda = (shard_ci + self.cfg.lambda_damp_factor * global_lambda_mean).reciprocal()
            # b, c = 12, 15

            # if self.rank == 0 and k == "layers.1.mlp.dense_h_to_4h" and rank_index == 0:
            #     self.logger.debug(f"{matrix_noi[0, b, c].item()}")
            #     self.logger.debug(f"{inverse_lambda[b, c].item()}")

            matrix_noi[:, start_row:end_row, :].mul_(inverse_lambda)
            # if self.rank == 0 and k == "layers.1.mlp.dense_h_to_4h" and rank_index == 0:
            #     self.logger.debug(f"{matrix_noi[0, b, c].item()}")
            # if self.rank == 0 and k == "layers.1.mlp.dense_h_to_4h" and rank_index == 0:
            #     self.logger.debug("--" * 50)
            if self.rank != rank_index:
                del shard_ci

    def _sharded_transpose_matmul(
        self,
        matrix_noi: Float[Tensor, "n o i"],
        matrix_bc: Float[Tensor, "b c"],
        mult_type: Literal["left", "right"],
    ):
        """
        Sharded matrix multiplication for distributed training.
        Assumes that c=i/world_size if left or o/world_size if right.
        gradients: [n, o, i]
        matrix_shard: [c, b] where b=i if left or b=o if right
        Returns: [n, o, c*w] if left or [n, c*w, i] if right
        """

        x, y = (
            (matrix_noi.shape[1], matrix_bc.shape[0] * self.world_size)
            if mult_type == "left"
            else (matrix_bc.shape[0] * self.world_size, matrix_noi.shape[2])
        )

        result_nxy = torch.zeros(matrix_noi.shape[0], x, y, device=matrix_noi.device, dtype=matrix_noi.dtype)

        for rank_index in range(self.world_size):
            if rank_index == self.rank:
                shard_bc = matrix_bc
            else:
                shard_bc = torch.zeros_like(matrix_bc)
            dist.broadcast(shard_bc, src=rank_index)

            shard_size = shard_bc.shape[0]
            start_row = rank_index * shard_size
            end_row = (rank_index + 1) * shard_size

            if mult_type == "left":
                result_nxy[:, :, start_row:end_row].copy_(torch.einsum("n o i, c i -> n o c", matrix_noi, shard_bc))
            elif mult_type == "right":
                result_nxy[:, start_row:end_row, :].copy_(torch.einsum("b o, n o i -> n b i", shard_bc, matrix_noi))
            if self.rank != rank_index:
                del shard_bc

        return result_nxy
