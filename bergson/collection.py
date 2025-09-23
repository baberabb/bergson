import json
import math
import os
import time
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
    GradientCollector,
    GradientProcessor,
)
from .peft import set_peft_enabled


class ETALogger:
    """Logs tqdm ETA over time to a file for plotting performance trends."""

    def __init__(self, log_file: str, total_batches: int):
        self.log_file = log_file
        self.total_batches = total_batches
        self.start_time = time.time()
        self.eta_data = []

    def log_eta(
        self,
        current_batch: int,
        eta_seconds: float,
        batch_time: float,
        memory_usage: float = 0.0,
    ):
        """Log current ETA and performance metrics."""
        elapsed_time = time.time() - self.start_time
        progress = current_batch / self.total_batches

        entry = {
            "timestamp": time.time(),
            "elapsed_time": elapsed_time,
            "current_batch": current_batch,
            "total_batches": self.total_batches,
            "progress": progress,
            "eta_seconds": eta_seconds,
            "eta_minutes": eta_seconds / 60,
            "batch_time": batch_time,
            "memory_usage_gb": memory_usage,
            "batches_per_second": 1.0 / batch_time if batch_time > 0 else 0,
        }

        self.eta_data.append(entry)

        # Write to file periodically (every 10 entries or at the end)
        if len(self.eta_data) % 10 == 0 or current_batch == self.total_batches - 1:
            self._write_to_file()

    def _write_to_file(self):
        """Write current data to the log file."""
        with open(self.log_file, "w") as f:
            json.dump(self.eta_data, f, indent=2)

    def get_performance_trends(self):
        """Analyze performance trends from logged data."""
        if len(self.eta_data) < 10:
            return {}

        # Calculate trends
        early_batches = self.eta_data[: len(self.eta_data) // 4]
        late_batches = self.eta_data[-len(self.eta_data) // 4 :]

        early_avg_time = sum(entry["batch_time"] for entry in early_batches) / len(
            early_batches
        )
        late_avg_time = sum(entry["batch_time"] for entry in late_batches) / len(
            late_batches
        )

        performance_degradation = (
            (late_avg_time - early_avg_time) / early_avg_time * 100
        )

        return {
            "performance_degradation_percent": performance_degradation,
            "early_avg_batch_time": early_avg_time,
            "late_avg_batch_time": late_avg_time,
            "total_entries": len(self.eta_data),
        }


def collect_gradients(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    path: str,
    *,
    batches: list[list[int]] | None = None,
    kl_divergence: bool | None = None,
    loss_reduction: Literal["mean", "sum"] = "mean",
    skip_preconditioners: bool = False,
    target_modules: set[str] | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # Batch size of one by default
    if batches is None:
        batches = [[idx] for idx in range(len(data))]

    # Mutable state for the GradientCollector callback
    mod_grads = {}
    preconditioners = {}

    # TODO: Handle this more elegantly
    dtype = torch.float32 if model.dtype == torch.float32 else torch.float16
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    lo = torch.finfo(dtype).min
    hi = torch.finfo(dtype).max

    def callback(name: str, g: torch.Tensor):
        g = g.flatten(1).clamp_(lo, hi)

        # Asynchronously move the gradient to CPU and convert to fp16
        mod_grads[name] = g.to(device="cpu", dtype=dtype, non_blocking=True)

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
    grad_sizes = {name: math.prod(s) for name, s in collector.shapes().items()}

    # Allocate structured space ahead of time for the gradients
    grad_buffer = create_index(
        path, num_grads=len(data), grad_sizes=grad_sizes, dtype=np_dtype
    )

    per_doc_losses = torch.full(
        (len(data),),
        device=model.device,
        dtype=dtype,
        fill_value=0.0,
    )

    # Performance tracking variables
    batch_times = []
    total_batches = len(batches)

    # Initialize ETA logger
    eta_logger = (
        ETALogger(os.path.join(path, "eta_log.json"), total_batches)
        if rank == 0
        else None
    )

    if rank == 0:
        print(f"Starting gradient collection with {total_batches} batches")
        print(f"Model dtype: {model.dtype}, Gradient dtype: {dtype}")
        print(f"Gradient buffer shape: {grad_buffer.shape}")
        print(f"ETA logging to: {os.path.join(path, 'eta_log.json')}")

    for batch_idx, indices in enumerate(
        tqdm(batches, disable=rank != 0, desc="Building index")
    ):
        batch_start_time = time.time()

        batch = data[indices]
        x, y = pad_and_tensor(
            batch["input_ids"],  # type: ignore
            labels=batch.get("labels"),  # type: ignore
            device=model.device,
        )
        masks = y[:, 1:] != -100
        denoms = masks.sum(dim=1, dtype=dtype) if loss_reduction == "mean" else 1.0

        if kl_divergence:
            with torch.inference_mode():
                set_peft_enabled(model, False)
                ref_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)
                set_peft_enabled(model, True)

            with collector:
                ft_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)

                # Compute average KL across all unmasked tokens
                kls = torch.sum(ft_lps.exp() * (ft_lps - ref_lps), dim=-1)
                losses = torch.sum(kls * masks, dim=-1) / denoms
                if "advantage" in batch:
                    losses *= torch.tensor(batch["advantage"], device=losses.device)

                losses.mean().backward()
        else:
            with collector:
                logits = model(x).logits[:, :-1]

                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y[:, 1:].flatten(),
                    reduction="none",
                ).reshape_as(y[:, 1:])
                losses = losses.sum(1) / denoms
                if "advantage" in batch:
                    losses *= torch.tensor(batch["advantage"], device=losses.device)

                losses.mean().backward()

        # Weirdly you need to explicitly synchronize here in order to make sure that
        # the nonblocking copies actually finish before we call .numpy()
        model.zero_grad()
        torch.cuda.synchronize()

        # It turns out that it's very important for efficiency to write the gradients
        # sequentially instead of first concatenating them, then writing to one vector
        for layer_name in mod_grads.keys():
            grad_buffer[layer_name][indices] = mod_grads[layer_name].numpy()

        mod_grads.clear()
        per_doc_losses[indices] = losses.detach().type_as(per_doc_losses)

        # Track performance metrics
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)

        # Log ETA and performance metrics
        if eta_logger:
            # Calculate ETA based on recent performance
            if len(batch_times) > 0:
                recent_avg_time = sum(batch_times[-min(10, len(batch_times)) :]) / min(
                    10, len(batch_times)
                )
                remaining_batches = total_batches - batch_idx - 1
                eta_seconds = remaining_batches * recent_avg_time
                eta_logger.log_eta(batch_idx, eta_seconds, batch_time)

        # Print diagnostics every 10% of batches or every 100 batches, whichever is smaller
        print_interval = max(1, min(100, total_batches // 10))
        if batch_idx % print_interval == 0 or batch_idx == total_batches - 1:
            if rank == 0:
                avg_time = sum(batch_times[-10:]) / min(
                    10, len(batch_times)
                )  # Last 10 batches
                print(
                    f"Batch {batch_idx}/{total_batches}: {batch_time:.3f}s (avg last 10: {avg_time:.3f}s)"
                )
    process_preconditioners(processor, preconditioners, len(data))

    if dist.is_initialized():
        dist.reduce(per_doc_losses, dst=0)

    if rank == 0:
        data = data.add_column(
            "loss",
            per_doc_losses.cpu().numpy(),
            feature=Value("float16" if dtype == torch.float16 else "float32"),
            new_fingerprint="loss",
        )
        data.save_to_disk(path + "/data.hf")

        processor.save(path)

    # Make sure the gradients are written to disk
    grad_buffer.flush()

    # Final performance summary and ETA analysis
    if rank == 0 and batch_times:
        total_time = sum(batch_times)
        avg_time = total_time / len(batch_times)
        print("\nPerformance Summary:")
        print(f"  Total batches: {len(batch_times)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per batch: {avg_time:.3f}s")
        print(
            f"  First 10 batches avg: {sum(batch_times[:10]) / min(10, len(batch_times)):.3f}s"
        )
        print(
            f"  Last 10 batches avg: {sum(batch_times[-10:]) / min(10, len(batch_times)):.3f}s"
        )

        if len(batch_times) > 20:
            # Check for performance degradation
            first_quarter = sum(batch_times[: len(batch_times) // 4]) / (
                len(batch_times) // 4
            )
            last_quarter = sum(batch_times[-len(batch_times) // 4 :]) / (
                len(batch_times) // 4
            )
            degradation = (last_quarter - first_quarter) / first_quarter * 100
            print(
                f"  Performance degradation: {degradation:.1f}% (first quarter vs last quarter)"
            )
            if degradation > 20:
                print("  WARNING: Significant performance degradation detected!")

        # ETA analysis
        if eta_logger:
            trends = eta_logger.get_performance_trends()
            if trends:
                print("\nETA Analysis:")
                print(
                    f"  Performance degradation: {trends['performance_degradation_percent']:.1f}%"
                )
                print(f"  Early avg batch time: {trends['early_avg_batch_time']:.3f}s")
                print(f"  Late avg batch time: {trends['late_avg_batch_time']:.3f}s")
                print(f"  Total ETA entries logged: {trends['total_entries']}")
                print(f"  ETA log saved to: {eta_logger.log_file}")


def process_preconditioners(
    processor: GradientProcessor,
    preconditioners: dict[str, torch.Tensor],
    len_data: int,
):
    """
    Aggregate preconditioners across ranks and compute their eigen decomposition
    distributed across all ranks.
    """

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    preconditioners_eigen = {}
    if rank == 0:
        print("Saving preconditioners...")
    for name, prec in preconditioners.items():
        if dist.is_initialized():
            dist.all_reduce(prec)

        preconditioners[name] = prec / len_data

    processor.preconditioners = preconditioners

    if rank == 0:
        print("Computing preconditioner eigen decompositions...")
    names = list(preconditioners.keys())
    names_per_rank = names[rank::world_size]

    for name in names_per_rank:
        original_dtype = preconditioners[name].dtype
        prec = preconditioners[name].to(dtype=torch.float64)
        eigvals, eigvecs = torch.linalg.eigh(prec)
        preconditioners_eigen[name] = (
            eigvals.to(dtype=original_dtype).contiguous(),
            eigvecs.to(dtype=original_dtype).contiguous(),
        )

    if rank == 0:
        print("Gathering and saving preconditioner eigen decompositions...")

    for name in names:
        prec = preconditioners[name]
        if name not in preconditioners_eigen:
            eigval = torch.zeros(prec.size(0), dtype=prec.dtype, device=prec.device)
            eigvec = torch.zeros_like(prec)
        else:
            eigval, eigvec = preconditioners_eigen[name]

        dist.all_reduce(eigval, op=dist.ReduceOp.SUM) if dist.is_initialized() else None
        dist.all_reduce(eigvec, op=dist.ReduceOp.SUM) if dist.is_initialized() else None

        preconditioners_eigen[name] = (eigval, eigvec)
    if rank == 0:
        processor.preconditioners_eigen = preconditioners_eigen
