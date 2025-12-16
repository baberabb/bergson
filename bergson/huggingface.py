import math
import os
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import Sized

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from numpy.typing import DTypeLike
from peft import PeftModel
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from bergson import AttentionConfig, GradientProcessor
from bergson.collector.gradient_collectors import StreamingGradientCollector
from bergson.data import create_index
from bergson.gradients import AdafactorNormalizer, AdamNormalizer
from bergson.utils.peft import detect_peft_modules


class GradientCollectorCallback(TrainerCallback):
    """Callback that collects gradients from the model during training.
    Implements the same functionality as `collect_gradients`."""

    def __init__(
        self,
        path: Path,
        attention_cfgs: dict[str, AttentionConfig] = {},
        projection_dim: int = 16,
        include_bias: bool = False,
        dtype: DTypeLike = np.float16,
        accumulate_grads: bool = False,
        use_optimizer_state: bool = True,
        track_order: bool = False,
    ):
        """
        Args:
            path: The path to save the gradients
            projection_dim: The dimension to project the gradients onto
            include_bias: Whether to append bias gradients when present on a module
            dtype: The dtype of the on-disk gradient store
            accumulate_grads: Whether to take the sum of the gradients
                of the same example across epochs. If `False`, the
                gradients for each epoch are stored separately.
            use_optimizer_state: Whether to use the optimizer state to
                normalize the gradients. If `False`, no normalization is
                applied.
            track_order: Whether to record the shuffled order of training data.
        attention_cfgs: Information used to split matrix-valued parameters into
            per-head matrices before down projection.
        """
        super().__init__()

        # Initialized in on_train_begin when we learn what the model is
        self.collector: StreamingGradientCollector
        self.grad_sizes = {}

        self.attention_cfgs = attention_cfgs
        self.accumulate_grads = accumulate_grads
        self.dtype = dtype
        self.path = path
        self.projection_dim = projection_dim
        self.include_bias = include_bias
        self.use_optimizer_state = use_optimizer_state
        self.order: list[dict] | None = [] if track_order else None

        self.eval_grad_buffers: dict[str, np.memmap] = {}
        self.eval_step_idxs: dict[str, int] = {}
        self.eval_indices: dict[str, list[int]] = {}

        self.mod_grads = {}
        self.batch_indices: Tensor | None = None

        # TODO: Handle this more elegantly
        self.torch_dtype = torch.float32 if self.dtype == np.float32 else torch.float16

    def write_grads(self, grad_buffer: np.memmap):
        torch.cuda.synchronize()
        for layer_name, g in self.collector.mod_grads.items():
            grad_buffer[layer_name][self.batch_indices, :] = g.numpy()

        self.collector.mod_grads.clear()

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        *,
        model: torch.nn.Module,
        **kwargs,
    ):
        if not hasattr(args, "__gradient_collection_enabled__"):
            raise RuntimeError(
                "Gradient collection is not enabled. Please enable it by "
                "calling bergson.prepare_gradient_collection on the trainer."
            )

        if isinstance(model, PeftModel):
            reshape_to_square = True
            target_modules = detect_peft_modules(model)  # type: ignore
        else:
            reshape_to_square = False
            target_modules = None

        self.collector = StreamingGradientCollector(
            model=getattr(model, "base_model", model),
            processor=GradientProcessor(
                {},
                projection_dim=self.projection_dim or None,
                reshape_to_square=reshape_to_square,
                include_bias=self.include_bias,
            ),
            target_modules=target_modules,
            attention_cfgs=self.attention_cfgs,
            dtype=self.torch_dtype,
        )
        self.grad_sizes = {
            name: math.prod(s) for name, s in self.collector.shapes().items()
        }

        # Record forward and backward hooks
        self.collector.__enter__()
        self.fwd_handle = model.register_forward_pre_hook(
            self.on_forward_begin,
            with_kwargs=True,
        )

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        *,
        eval_dataloader: DataLoader | dict[str, DataLoader],
        train_dataloader: DataLoader,
        **kwargs,
    ):
        epoch = int(state.epoch or 0)

        if self.accumulate_grads and epoch > 0:
            return

        epoch_suffix = "" if self.accumulate_grads else f"/epoch_{epoch}"

        ds = train_dataloader.dataset
        if not isinstance(ds, Sized):
            raise ValueError("Dataset must be sized for gradient collection")

        self.train_grad_buffer = create_index(
            self.path / ("train" + epoch_suffix),
            num_grads=len(ds),
            grad_sizes=self.grad_sizes,
            dtype=self.dtype,
        )
        self.train_step_idx = 0

        # Set up the gradient buffers for the evaluation datasets
        if eval_dataloader is None:
            return
        elif isinstance(eval_dataloader, dict):
            eval_datasets = eval_dataloader
        else:
            eval_datasets = {"eval": eval_dataloader}

        for dataset_name, dataloader in eval_datasets.items():
            self.eval_grad_buffers[dataset_name] = create_index(
                self.path / (dataset_name + epoch_suffix),
                num_grads=len(dataloader),
                grad_sizes=self.grad_sizes,
                dtype=self.dtype,
            )
            self.eval_step_idxs[dataset_name] = 0
            self.eval_indices[dataset_name] = [
                i.item() for batch in dataloader for i in batch["_idx"]
            ]

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            epoch = int(state.epoch or 0) - 1
            epoch_suffix = "" if self.accumulate_grads else f"/epoch_{epoch}"
            path = self.path / ("train" + epoch_suffix)

            assert self.collector is not None
            self.collector.processor.save(path)

        # Ensure the gradients are written to disk
        self.train_grad_buffer.flush()
        for eval_grad_buffer in self.eval_grad_buffers.values():
            eval_grad_buffer.flush()

    def on_forward_begin(self, _: torch.nn.Module, args, kwargs: dict):
        # Record the original indices of this batch
        self.batch_indices = kwargs.pop("_idx").to("cpu", non_blocking=True)
        return args, kwargs

    def on_module_backward(self, name: str, g: Tensor):
        lo = torch.finfo(self.torch_dtype).min
        hi = torch.finfo(self.torch_dtype).max
        g = g.flatten(1).clamp_(lo, hi)

        # Asynchronously move the gradient to CPU and convert to fp16
        self.mod_grads[name] = g.to(
            device="cpu", dtype=self.torch_dtype, non_blocking=True
        )

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each training step.
        If using gradient accumulation, one training step might take several inputs."""
        self.write_grads(self.train_grad_buffer)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ):
        self.on_substep_end(args, state, control)
        print("Step end")

        # Record training order if enabled
        if self.order is not None:
            assert (
                self.batch_indices is not None
            ), "Batch indices are not available for training order tracking"

            epoch = int(state.epoch or 0)
            global_step = state.global_step

            self.order.extend(
                {
                    "_idx": int(idx),
                    # global_step is 1-indexed.
                    "global_step": global_step,
                    "epoch": epoch,
                }
                for idx in self.batch_indices.tolist()
            )

        # We can skip all this if we're not using the optimizer state
        if not self.use_optimizer_state:
            return

        # The optimizer doesn't actually know the names of the parameters
        model = getattr(model, "base_model", model)
        param_to_name = {
            param: name
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        normalizers: dict[str, AdafactorNormalizer] = {}

        assert self.collector is not None
        proc = self.collector.processor
        proc.normalizers = {}

        # Read normalizers off of the optimizer state. We need to figure out
        # what type of optimizer this is first.
        for group in optimizer.param_groups:
            lr_sqrt = group["lr"] ** 0.5

            for param in group["params"]:
                name = param_to_name[param].removesuffix(".weight")
                if name not in self.collector.target_info:
                    continue

                p_state = optimizer.state[param]

                # Adam-like optimizer
                if (eas := p_state.get("exp_avg_sq")) is not None:
                    norm = AdamNormalizer(eas).to_adafactor()

                # Adafactor-like optimizer
                elif (vr := p_state.get("exp_avg_sq_row")) is not None:
                    vc = p_state.get("exp_avg_sq_col")
                    norm = AdafactorNormalizer(vr, vc)
                else:
                    continue

                # Scale the gradient by the current learning rate. It's factorized
                # so we multiply each factor by the square root of the LR.
                norm.row *= lr_sqrt
                norm.col *= lr_sqrt
                normalizers[name] = norm

        proc.normalizers = normalizers

    def on_prediction_step(self, args, state, control, **kwargs):
        dataset_name = kwargs["inputs"]["dataset_name"]
        self.write_grads(self.eval_grad_buffers[dataset_name])

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        assert self.collector is not None
        self.collector.__exit__(None, None, None)
        self.fwd_handle.remove()

        if self.order is not None:
            self._save_order()

    def _save_order(self):
        """Save the training order to disk, handling distributed training."""
        assert self.order is not None
        os.makedirs(self.path, exist_ok=True)

        if dist.is_initialized():
            # Gather training order from all processes
            all_orders = [None] * dist.get_world_size()
            dist.all_gather_object(all_orders, self.order)

            # Only rank 0 saves the merged data
            if dist.get_rank() == 0:
                merged_order = list(
                    chain.from_iterable(
                        order for order in all_orders if order is not None
                    )
                )
                dataset = Dataset.from_list(merged_order)
                dataset.save_to_disk(os.path.join(self.path, "order.hf"))

        else:
            dataset = Dataset.from_list(self.order)
            dataset.save_to_disk(os.path.join(self.path, "order.hf"))


def prepare_for_gradient_collection(trainer: Trainer):
    """Mutate the trainer and its datasets in-place to expose the datasets'
    indices to the gradient collector callback."""
    # Add indices to the training dataset
    trainer.train_dataset = trainer.train_dataset.map(  # type: ignore
        lambda ex, idx: {"_idx": idx}, with_indices=True
    )

    # Add indices to the evaluation dataset/s if they exist
    if trainer.eval_dataset is not None:
        if isinstance(trainer.eval_dataset, dict):
            for eval_name, dataset in trainer.eval_dataset.items():
                trainer.eval_dataset[eval_name] = dataset.map(  # type: ignore
                    lambda ex, idx: {"_idx": idx}, with_indices=True
                )
        else:
            trainer.eval_dataset = trainer.eval_dataset.map(  # type: ignore
                lambda ex, idx: {"_idx": idx}, with_indices=True
            )

    trainer._set_signature_columns_if_needed()
    trainer._signature_columns.append("_idx")  # type: ignore

    if trainer.data_collator:
        original_collator = trainer.data_collator

        @wraps(original_collator)  # type: ignore
        def wrapped_collator(features):
            batch = original_collator(features)
            batch.setdefault("_idx", torch.tensor([f["_idx"] for f in features]))
            return batch

        trainer.data_collator = wrapped_collator

    trainer.args.__gradient_collection_enabled__ = True  # type: ignore

    return trainer
