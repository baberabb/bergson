import math
from typing import Literal

# import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, Value
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .data import Query, create_index, pad_and_tensor
from .gradients import AttentionConfig, GradientCollector, GradientProcessor
from .peft import set_peft_enabled


def collect_gradients(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    path: str,
    token_batch_size: int,
    *,
    batches: list[list[int]] | None = None,
    kl_divergence: bool | None = None,
    loss_reduction: Literal["mean", "sum"] = "mean",
    skip_preconditioners: bool = False,
    target_modules: set[str] | None = None,
    attention_cfgs: dict[str, AttentionConfig] | None = None,
    save_index: bool = True,
    save_processor: bool = True,
    drop_columns: bool = False,
    query: Query | None = None,
    module_wise: bool = False,
    create_custom_query: bool = False,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    assert not create_custom_query, "create_custom_query is commented out"
    if module_wise and query:
        assert skip_preconditioners

    rank = dist.get_rank() if dist.is_initialized() else 0

    if attention_cfgs is None:
        attention_cfgs = {}

    # Batch size of one by default
    if batches is None:
        batches = [[idx] for idx in range(len(data))]

    # Mutable state for the GradientCollector callback
    mod_grads = {}
    preconditioners = processor.preconditioners

    # TODO: Handle this more elegantly
    dtype = torch.float32 if model.dtype == torch.float32 else torch.float16
    np_dtype = np.float32 if dtype == torch.float32 else np.float16
    lo = torch.finfo(dtype).min
    hi = torch.finfo(dtype).max

    collector = GradientCollector(
        model.base_model,
        lambda _: None,
        processor,
        target_modules=target_modules,
        attention_cfgs=attention_cfgs,
    )

    # Allocate space ahead of time for the gradients
    grad_sizes = {name: math.prod(s) for name, s in collector.shapes().items()}

    # Allocate structured space ahead of time for the gradients
    grad_buffer = (
        create_index(path, num_grads=len(data), grad_sizes=grad_sizes, dtype=np_dtype)
        if save_index
        else None
    )
    # if create_custom_query:
    #     num_grads = sum(len(indices) for indices in batches)
    #     print("file size in GB", sum(list(grad_sizes.values())) *
    # np.dtype(np_dtype).itemsize / 1024**3)
    #     grads = {
    #         name: torch.zeros(1, grad_sizes[name], dtype=torch.float16, device="cpu")
    #         for name in grad_sizes.keys()
    #     }
    # else:
    #     grads = {}
    #     num_grads = -1

    def callback(name: str, g: torch.Tensor, indices: list[int]):
        g = g.flatten(1).clamp_(lo, hi)
        if grad_buffer is not None:  # or grads:
            # Asynchronously move the gradient to CPU and convert to the final dtype
            mod_grads[name] = g.to(device="cpu", dtype=dtype, non_blocking=True)

            if module_wise:
                # Consume gradient immediately
                torch.cuda.synchronize()
                # if grads:
                #     grads[name][0, :] += mod_grads[name].sum(dim=0) / len(indices)
                # elif grad_buffer is not None:
                if grad_buffer is not None:
                    grad_buffer[name][indices] = mod_grads[name].numpy()

                mod_grads.pop(name)
        else:
            # TODO do we need the dtype conversion
            mod_grads[name] = g.to(dtype=dtype)
            if module_wise and query:
                query(indices, mod_grads, name)
                mod_grads.pop(name)

        # Compute the outer product of the flattened gradient
        if not skip_preconditioners:
            g = g.float()
            preconditioner = preconditioners.get(name, None)
            if preconditioner is None:
                preconditioners[name] = g.mT @ g
            else:
                preconditioner.addmm_(g.mT, g)

    # Update collect with callback
    collector.closure = callback

    # Run a random tensor of size token_batch_size to warm up the cache
    # And ensure enough memory is available
    random_tensor = torch.randint(
        0, 50_000, (1, token_batch_size), device=model.device, dtype=torch.long
    )
    with GradientCollector(
        model.base_model,
        lambda _: None,
        processor,
        target_modules=target_modules,
        attention_cfgs=attention_cfgs,
    ) as collector:
        logits = model(random_tensor).logits[:, :-1]
        loss = logits[0, 0, 0].float()
        model.zero_grad()
        del random_tensor, logits, loss
        torch.cuda.synchronize()
        # Optional: trim the caching allocator between very different shapes
        torch.cuda.empty_cache()

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
        denoms = masks.sum(dim=1, dtype=dtype) if loss_reduction == "mean" else 1.0

        if kl_divergence:
            with torch.inference_mode():
                set_peft_enabled(model, False)
                ref_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)
                set_peft_enabled(model, True)

            with GradientCollector(
                model.base_model,
                callback,
                processor,
                target_modules=target_modules,
                attention_cfgs=attention_cfgs,
                indices=indices,
            ) as collector:
                ft_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)

                # Compute average KL across all unmasked tokens
                kls = torch.sum(ft_lps.exp() * (ft_lps - ref_lps), dim=-1)
                losses = torch.sum(kls * masks, dim=-1) / denoms
                if "advantage" in batch:
                    losses *= torch.tensor(batch["advantage"], device=losses.device)

                losses.mean().backward()
        else:
            with GradientCollector(
                model.base_model,
                callback,
                processor,
                target_modules=target_modules,
                attention_cfgs=attention_cfgs,
                indices=indices,
            ) as collector:
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

        if grad_buffer is not None and not module_wise:
            # Weirdly you need to explicitly synchronize here in order to make
            # sure that the nonblocking copies actually finish before we call
            # .numpy()
            torch.cuda.synchronize()

            # It turns out that it's very important for efficiency to write the
            # gradients sequentially instead of first concatenating them, then
            # writing to one vector
            # if custom:
            # for name in mod_grads.keys():
            # grads[name] += mod_grads[name].sum(dim=0) / len(indices)
            # else:
            for module_name in mod_grads.keys():
                grad_buffer[module_name][indices] = mod_grads[module_name].numpy()

        if query and not module_wise:
            query(indices, mod_grads)

        mod_grads.clear()
        per_doc_losses[indices] = losses.detach().type_as(per_doc_losses)

    # TODO can probably loosen this condition
    if save_processor or not skip_preconditioners:
        process_preconditioners(processor, preconditioners, len(data))

    if dist.is_initialized():
        dist.reduce(per_doc_losses, dst=0)

    if rank == 0:
        if drop_columns:
            data = data.remove_columns(["input_ids"])

        data = data.add_column(
            "loss",
            per_doc_losses.cpu().numpy(),
            feature=Value("float16" if dtype == torch.float16 else "float32"),
            new_fingerprint="loss",
        )
        data.save_to_disk(path + "/data.hf")

        if save_processor:
            processor.save(path)

    # Make sure the gradients are written to disk
    if grad_buffer is not None:
        grad_buffer.flush()

    # if create_custom_query:
    #     torch.save(grads, os.path.join(path, f"accum_mean_grads_{rank}.pth"))
    #     torch.save(num_grads, os.path.join(path, f"num_grads_{rank}.pth"))


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
