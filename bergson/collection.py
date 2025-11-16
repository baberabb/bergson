import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, Value
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from bergson.score.scorer import Scorer

from .data import IndexConfig, ReduceConfig, create_index, pad_and_tensor
from .gradients import AttentionConfig, GradientCollector, GradientProcessor
from .peft import set_peft_enabled


class Builder:
    num_items: int
    grad_buffer: np.memmap
    reduce_cfg: ReduceConfig | None

    def __init__(
        self,
        path: Path,
        data: Dataset,
        grad_sizes: dict[str, int],
        dtype: torch.dtype,
        reduce_cfg: ReduceConfig | None = None,
    ):
        self.grad_sizes = grad_sizes
        self.num_items = len(data)
        self.reduce_cfg = reduce_cfg

        num_grads = self.num_items if self.reduce_cfg is None else 1

        self.in_memory_grad_buffer = (
            np.zeros((1, sum(self.grad_sizes.values())), dtype=np.float32)
            if self.reduce_cfg is None
            else None
        )

        # TODO: Handle this more elegantly
        np_dtype = np.float32 if dtype == torch.float32 else np.float16

        self.grad_buffer = create_index(
            path,
            num_grads=num_grads,
            grad_sizes=self.grad_sizes,
            dtype=np_dtype,
            with_structure=False,
        )

    def acc_grads(self, indices: list[int], mod_grads: dict[str, torch.Tensor]):
        assert self.reduce_cfg is not None and self.in_memory_grad_buffer is not None

        if self.reduce_cfg.unit_normalize:
            ssqs = torch.zeros(len(indices))
            for module_name in self.grad_sizes.keys():
                ssqs += mod_grads[module_name].pow(2).sum(dim=0)
            norms = ssqs.sqrt()
        else:
            norms = torch.ones(len(indices))

        offset = 0
        for module_name in self.grad_sizes.keys():
            mod_grads[module_name] /= norms
            self.in_memory_grad_buffer[
                0, offset : offset + mod_grads[module_name].shape[1]
            ] += (mod_grads[module_name].sum(dim=0).numpy().astype(np.float32)) / norms
            offset += mod_grads[module_name].shape[1]

    def __call__(self, indices: list[int], mod_grads: dict[str, torch.Tensor]):
        torch.cuda.synchronize()

        if self.reduce_cfg is not None:
            self.acc_grads(indices, mod_grads)

        # It turns out that it's very important for efficiency to write the
        # gradients sequentially instead of first concatenating them, then
        # writing to one vector
        offset = 0
        for module_name in self.grad_sizes.keys():
            self.grad_buffer[
                indices, offset : offset + mod_grads[module_name].shape[1]
            ] = mod_grads[module_name].numpy()
            offset += mod_grads[module_name].shape[1]

    def flush(self):
        self.grad_buffer.flush()

    def all_reduce(self):
        if self.reduce_cfg is None:
            return

        assert self.in_memory_grad_buffer is not None

        dist.all_reduce(self.in_memory_grad_buffer, op=dist.ReduceOp.SUM)

        if self.reduce_cfg.method == "mean":
            self.in_memory_grad_buffer /= self.num_items

        self.grad_buffer[:] = self.in_memory_grad_buffer.copy()
        self.flush()


def collect_gradients(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    cfg: IndexConfig,
    *,
    batches: list[list[int]] | None = None,
    target_modules: set[str] | None = None,
    attention_cfgs: dict[str, AttentionConfig] | None = None,
    scorer: Scorer | None = None,
    reduce_cfg: ReduceConfig | None = None,
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
    preconditioners = processor.preconditioners

    # TODO: Handle this more elegantly
    dtype = torch.float32 if model.dtype == torch.float32 else torch.float16
    lo = torch.finfo(dtype).min
    hi = torch.finfo(dtype).max

    def callback(name: str, g: torch.Tensor):
        g = g.flatten(1).clamp_(lo, hi)
        if cfg.save_index:
            # Asynchronously move the gradient to CPU and convert to the final dtype
            mod_grads[name] = g.to(device="cpu", dtype=dtype, non_blocking=True)
        else:
            mod_grads[name] = g.to(dtype=dtype)

        # Compute the outer product of the flattened gradient
        if not cfg.skip_preconditioners:
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
        attention_cfgs=attention_cfgs or {},
    )

    # Allocate space ahead of time for the gradients
    grad_sizes = {name: math.prod(s) for name, s in collector.shapes().items()}
    builder = (
        Builder(cfg.partial_run_path, data, grad_sizes, dtype, reduce_cfg)
        if cfg.save_index
        else None
    )

    per_doc_losses = torch.full(
        (len(data),),
        device=model.device,
        dtype=dtype,
        fill_value=0.0,
    )

    for indices in tqdm(batches, disable=rank != 0, desc="Building index"):
        batch = data[indices]
        x, y = pad_and_tensor(
            batch["input_ids"],  # type: ignore
            labels=batch.get("labels"),  # type: ignore
            device=model.device,
        )
        masks = y[:, 1:] != -100
        denoms = masks.sum(dim=1, dtype=dtype) if cfg.loss_reduction == "mean" else 1.0

        if cfg.loss_fn == "kl":
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

        model.zero_grad()

        if builder is not None:
            builder(indices, mod_grads)

        if scorer is not None:
            scorer(indices, mod_grads)

        mod_grads.clear()
        per_doc_losses[indices] = losses.detach().type_as(per_doc_losses)

    process_preconditioners(processor, preconditioners, len(data))

    if dist.is_initialized():
        dist.reduce(per_doc_losses, dst=0)

    if rank == 0:
        if cfg.drop_columns:
            data = data.remove_columns(["input_ids"])

        data = data.add_column(
            "loss",
            per_doc_losses.cpu().numpy(),
            feature=Value("float16" if dtype == torch.float16 else "float32"),
            new_fingerprint="loss",
        )

        data.save_to_disk(cfg.partial_run_path / "data.hf")

        processor.save(cfg.partial_run_path)

    # Make sure the gradients are written to disk
    if builder is not None:
        builder.flush()
        builder.all_reduce()


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
