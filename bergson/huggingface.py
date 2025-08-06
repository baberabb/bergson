import os
from copy import deepcopy
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist
from numpy.typing import DTypeLike
from torch import Tensor
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from bergson import GradientCollector, GradientProcessor
from bergson.data import create_index


class GradientCollectorCallback(TrainerCallback):
    """Callback that collects gradients from the model during training.
    Implements the same functionality as `collect_gradients`."""

    def __init__(
        self,
        collector: GradientCollector,
        processor: GradientProcessor,
        mod_grads: dict[str, Tensor],
        preconditioners: dict[str, Tensor],
        grad_sizes: dict[str, int],
        path: str,
        dtype: DTypeLike = np.float16,
        mean_gradients: bool = False,
        normalize_eval: bool = False,
    ):
        """
        Args:
            grad_sizes: The sizes of the module gradients
            processor: The gradient processor
            mod_grads: The module gradients
            preconditioners: The gradient preconditioners
            path: The path to the gradient store
            dtype: The dtype of the on-disk gradient store
            mean_gradients: Whether to take the mean of the gradients
                of the same example across epochs. If `False`, the
                gradients for each epoch are stored separately.
            normalize_eval: Whether to normalize the evaluation gradients
                using the training optimizer normalization.
        """
        super().__init__()
        self.collector = collector
        self.processor = processor
        self.mod_grads = mod_grads
        self.preconditioners = preconditioners
        self.grad_sizes = grad_sizes
        self.path = path
        self.dtype = dtype
        self.mean_gradients = mean_gradients
        self.normalize_eval = normalize_eval

        self.eval_grad_buffers: dict[str, np.memmap] = {}
        self.eval_step_idxs: dict[str, int] = {}
        self.eval_indices: dict[str, list[int]] = {}

    def write_grads(
        self,
        grad_buffer: np.memmap,
        mod_grads: dict[str, Tensor],
        indices: list[int],
    ):
        for layer_name in mod_grads.keys():
            torch.cuda.synchronize()
            grad_buffer[layer_name][indices] += mod_grads[layer_name].numpy() / self.div
        self.mod_grads.clear()

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        assert hasattr(args, "__gradient_collection_enabled__"), (
            "Gradient collection is not enabled. Please enable it by "
            "calling bergson.prepare_gradient_collection on the trainer."
        )
        self.div = args.num_train_epochs if self.mean_gradients else 1

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        epoch = int(state.epoch or 0)

        if self.mean_gradients and epoch > 0:
            return

        epoch_suffix = "" if self.mean_gradients else f"/epoch_{epoch}"

        if dist.is_initialized():
            num_examples = torch.tensor(
                [len(kwargs["train_dataloader"].dataset)],
                device="cuda",
                dtype=torch.int32,
            )
            dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
            self.num_examples = int(num_examples.item())
        else:
            self.num_examples = len(kwargs["train_dataloader"].dataset)

        # Set up the gradient buffers for the training dataset
        self.train_indices = [
            i.item() for batch in kwargs["train_dataloader"] for i in batch["_idx"]
        ]
        self.train_grad_buffer = create_index(
            os.path.join(self.path, "train" + epoch_suffix),
            num_grads=self.num_examples,
            grad_sizes=self.grad_sizes,
            dtype=self.dtype,
        )
        self.train_step_idx = 0

        # Set up the gradient buffers for the evaluation datasets
        if kwargs["eval_dataloader"] is None:
            return
        elif isinstance(kwargs["eval_dataloader"], dict):
            eval_datasets = kwargs["eval_dataloader"]
        else:
            eval_datasets = {"eval": kwargs["eval_dataloader"]}

        for dataset_name, dataloader in eval_datasets.items():
            self.eval_grad_buffers[dataset_name] = create_index(
                os.path.join(self.path, dataset_name + epoch_suffix),
                num_grads=len(dataloader),
                grad_sizes=self.grad_sizes,
                dtype=self.dtype,
            )
            self.eval_step_idxs[dataset_name] = 0
            self.eval_indices[dataset_name] = [
                i.item() for batch in dataloader for i in batch["_idx"]
            ]

        if epoch > 0 and not self.normalize_eval:
            self.collector.processor = self.processor

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Save the preconditioners
        chols = {}
        for name, prec in self.preconditioners.items():
            if dist.is_initialized():
                dist.all_reduce(prec)

            L, info = torch.linalg.cholesky_ex(prec / self.num_examples)
            if info.any() and rank == 0:
                print(f"Warning: {name} has a singular second moment matrix.")

            chols[name] = L

        self.processor.preconditioners = chols

        if rank == 0:
            self.processor.save(self.path)

        # Ensure the gradients are written to disk
        self.train_grad_buffer.flush()
        for eval_grad_buffer in self.eval_grad_buffers.values():
            eval_grad_buffer.flush()

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each training step.
        If using gradient accumulation, one training step might take several inputs."""
        indices = self.train_indices[
            self.train_step_idx : self.train_step_idx + args.per_device_train_batch_size
        ]
        self.train_step_idx += args.per_device_train_batch_size

        self.write_grads(self.train_grad_buffer, self.mod_grads, indices)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self.normalize_eval:
            eval_processor = deepcopy(self.collector.processor)
            eval_processor.normalizers = {}
            self.collector.processor = eval_processor

    def on_prediction_step(self, args, state, control, **kwargs):
        dataset_name = kwargs["inputs"]["dataset_name"]

        indices = self.eval_indices[dataset_name][
            self.eval_step_idxs[dataset_name] : self.eval_step_idxs[dataset_name]
            + args.per_device_eval_batch_size
        ]

        self.eval_step_idxs[dataset_name] += args.per_device_eval_batch_size

        self.write_grads(
            self.eval_grad_buffers[dataset_name],
            self.mod_grads,
            indices,
        )


def prepare_for_gradient_collection(trainer: Trainer):
    """Mutate the trainer object and its datasets in-place to expose the dataset
    indices to the gradient collector callback."""
    # Add indices to the training dataset
    trainer.train_dataset = trainer.train_dataset.map(
        lambda ex, idx: {"_idx": idx}, with_indices=True
    )

    # Add indices to the evaluation dataset/s if they exist
    if trainer.eval_dataset is not None:
        if isinstance(trainer.eval_dataset, dict):
            for eval_name, dataset in trainer.eval_dataset.items():
                trainer.eval_dataset[eval_name] = dataset.map(
                    lambda ex, idx: {"_idx": idx}, with_indices=True
                )
        else:
            trainer.eval_dataset = trainer.eval_dataset.map(
                lambda ex, idx: {"_idx": idx}, with_indices=True
            )

    # Mutate the trainer to retain the indices
    def retain_idx(collator):
        @wraps(collator)
        def wrapper(features):
            batch = collator(features)
            batch.setdefault("_idx", torch.tensor([f["_idx"] for f in features]))
            return batch

        return wrapper

    trainer.data_collator = retain_idx(trainer.data_collator)
    trainer.args.remove_unused_columns = False
    trainer.args.__gradient_collection_enabled__ = True  # type: ignore

    return trainer
