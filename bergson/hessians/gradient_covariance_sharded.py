import json
import os
import random
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import send_to_device
from datasets import Dataset
from safetensors.torch import save_file
from torch import Tensor, autocast
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, default_data_collator
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from bergson.approx_unrolling.language_task import LanguageModelingTask, Task
from bergson.approx_unrolling.model_checkpoints import PythiaCheckpoints
from bergson.approx_unrolling.pile_data import get_pile_dataset, randomize_batch
from bergson.approx_unrolling.utils import TensorDict
from bergson.data import MemmapDataset
from bergson.gradients import Normalizer

NORMALIZER_TYPES: dict[str, type["Normalizer"]] = {}

NORMALIZER_TYPES: dict[str, type["Normalizer"]] = {}


def setup_distributed():
    """Properly initialize distributed training"""
    try:
        # Check if we're in a distributed environment
        if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Initialize the process group first
            dist.init_process_group(backend="nccl")

            # Set CUDA device for current rank
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            print("Distributed training initialized successfully")
            return dist.get_rank(), dist.get_world_size()
        else:
            # Not in distributed environment
            raise RuntimeError("Not in distributed environment")
    except:
        # Fall back to single GPU
        print("Falling back to single GPU")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1


@dataclass
class CovarianceProcessor:
    """Configuration for processing and compressing gradients."""

    task: Task
    """
    The task in question, in particular defines the training loss function
    """

    gradient_covariances: Mapping[str, Tensor] = field(default_factory=dict)
    """
    Dictionary of preconditioners for each matrix-valued parameter in the model.
    These are applied after the normalization and random projection steps.
    """

    activation_covariance: Mapping[str, Tensor] = field(default_factory=dict)
    """
    Dictionary of preconditioners for each matrix-valued parameter in the model.
    These are applied after the normalization and random projection steps.
    """

    sampling_seed: int = 42
    """Seed for generating the random projection matrices."""

    def compute_covariances(
        self,
        model: nn.Module,
        data: Dataset | MemmapDataset,
    ):
        """
        Estimate preconditioners from data. Overwrites the `preconditioners` field.
        """

        rank = dist.get_rank() if dist.is_initialized() else 0

        if hasattr(model, "device"):
            model_device = model.device
        else:
            # For FSDP models, get device from parameters
            model_device = next(model.parameters()).device

        if rank == 0:
            print(f"Model device: {model_device}")

        if dist.is_initialized() and world_size > 1:
            sampler = DistributedSampler(data, num_replicas=world_size, rank=rank)  # type: ignore
            batch_size_per_gpu = 32 // world_size  # Adjust batch size for distributed
        else:
            sampler = SequentialSampler(data)
            batch_size_per_gpu = 32

        # TODO: Double check this
        dataloader = DataLoader(
            data,  # type: ignore
            batch_size=batch_size_per_gpu,
            sampler=sampler,
            collate_fn=default_data_collator,
            pin_memory=True,  # Better performance
            num_workers=2,  # Parallel data loading
        )

        activation_covariances = {}
        gradient_covariances = {}

        def callback_activation(name: str, a: torch.Tensor):
            activation_covariance = activation_covariances.get(name, None)
            if activation_covariance is None:
                activation_covariances[name] = a.T.matmul(a)
            else:
                activation_covariance.addmm_(a.T, a)

        def callback_gradient(name: str, g: torch.Tensor):
            gradient_covariance = gradient_covariances.get(name, None)

            g = g.reshape(-1, g.shape[-1])  # [N*S, O]

            if gradient_covariance is None:
                # Initialize the covariance matrix for this module
                gradient_covariances[name] = g.T.matmul(g)
            else:
                gradient_covariances[name].addmm_(g.T, g)  # [O,O]

        for batch in tqdm(dataloader, position=rank):
            batch = send_to_device(batch, model_device)
            randomize_batch(batch, rank)
            print(f"Processing batch on rank {rank}: {batch.keys()}")
            print(f"Batch shapes: {[v.shape for v in batch.values()]}")
            with GradientCollector(
                model, self, activation_closure=callback_activation, gradient_closure=callback_gradient
            ) as mgr:
                model.zero_grad()
                with autocast(
                    device_type="cuda",
                    enabled=True,
                    dtype=torch.float32,
                ):
                    loss = self.task.compute_train_loss(batch=batch, model=model, sample=False)

                loss.backward()

            del loss

        # Reduce the preconditioners across processes if needed
        if dist.is_initialized():
            for activation_covariance in activation_covariances.values():
                dist.all_reduce(activation_covariance)

            for gradient_covariance in gradient_covariances.values():
                dist.all_reduce(gradient_covariance)

        # save using safetensors
        save_dir = "influence_results_sharded"
        os.makedirs(save_dir, exist_ok=True)
        save_file(activation_covariances, os.path.join(save_dir, "activation_covariance.safetensors"))
        save_file(gradient_covariances, os.path.join(save_dir, "gradient_covariance.safetensors"))

        if dist.is_initialized():
            dist.destroy_process_group()
        print("saved")

    @classmethod
    def load(
        cls,
        path: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "CovarianceProcessor":
        """
        Load the normalizers and preconditioners from a file.
        """
        cfg_path = os.path.join(path, "processor_config.json")
        norm_path = os.path.join(path, "normalizers.pth")
        gradient_covariance_path = os.path.join(path, "gradient_covariance.safetensors")
        activation_covariance_path = os.path.join(path, "activation_covariance.safetensors")

        # Load configuration
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Load normalizers
        norm_state = torch.load(
            norm_path,
            map_location=map_location,
            weights_only=True,
        )
        normalizers = {name: Normalizer.from_state_dict(state) for name, state in norm_state.items()}
        # TODO: Fix
        return cls(  # type:ignore
            normalizers=normalizers,  # type:ignore
            gradient_covariance=torch.load(  # type:ignore
                gradient_covariance_path,
                map_location=map_location,
                weights_only=True,
            ),
            activation_covariance=torch.load(
                activation_covariance_path,
                map_location=map_location,
                weights_only=True,
            ),
            sampling_seed=cfg.get("sampling_seed", 42),
        )

    def save(self, path: str):
        """
        Save the normalizers and preconditioners to a file.
        """
        os.makedirs(path, exist_ok=True)

        cfg_path = os.path.join(path, "processor_config.json")
        norm_path = os.path.join(path, "normalizers.pth")
        precond_path = os.path.join(path, "preconditioners.pth")


@dataclass
class GradientCollector(ContextDecorator):
    """
    Adds forward and backward hooks to `model` that efficiently collect per-sequence
    gradients for all the matrix-valued parameters, randomly projecting them using a
    fixed seed to compress them into lower-dimensional blocks of shape [p×q]. We use
    a dictionary of `AdafactorNormalizer` to scale the gradients by the second moments
    of the parameters, which are expected to be precomputed and passed in.

    The collected gradients are flattened into a single tensor after the backward pass.
    You can access the flattened gradients via the `flat_grads` attribute after exiting
    the context manager.

    We assume that the input to `model` is of shape `[N, S, I]`, where `N` is the
    batch size, `S` is the sequence length, and `I` is the input dimension. We take the
    mean over the sequence length to obtain a single gradient per sequence.
    """

    model: nn.Module

    processor: CovarianceProcessor
    """Configuration for processing and compressing gradients."""

    activation_closure: Callable | None = None
    """Closure to call on the activation as it is collected. If provided, we will not
    store the activation after the closure is called."""

    gradient_closure: Callable | None = None
    """Closure to call on the gradient as it is collected. If provided, we will not
    store the gradient after the closure is called."""

    def __post_init__(self):
        self._fwd_hooks: list[RemovableHandle] = []
        self._bwd_hooks: list[RemovableHandle] = []

        # We actually take advantage of the fact that modern Python dicts are ordered
        # so that we can both keep track of the order in which the hooks are called
        # and also use the names of the layers as keys for the normalizers.
        self.grad_covariance: dict[str, Tensor] = {}
        self.activation_covariance: dict[str, Tensor] = {}

    def __enter__(self):
        # install a hook on every Linear
        for name, layer in self.model.named_modules():
            if not isinstance(layer, nn.Linear):
                continue

            if "embed" in name:
                continue

            # Save the name of the layer for later use
            layer._name = name  # type: ignore[attr-defined]

            # register forward hook to save V = X @ B^T
            fwd_hook = layer.register_forward_hook(self._save_input)
            self._fwd_hooks.append(fwd_hook)

            # register backward hook to compute P = sum(U @ V^T)
            bwd_hook = layer.register_full_backward_hook(self._process_grad)
            self._bwd_hooks.append(bwd_hook)

        return self

    def _save_input(self, module: nn.Module, inp: tuple, _):
        """Save the input to the module for later use in the backward pass."""
        x = inp[0].detach()
        assert isinstance(x, Tensor)
        assert x.ndim == 3, f"Expected input of shape [N, S, I], got {x.shape}"

        A = x.reshape(-1, x.shape[-1])  # [N*S,O]

        if module.bias is not None:
            append_term = A.new_ones((A.size(0), 1), requires_grad=False)
            A = torch.cat([A, append_term], dim=-1)

        if self.activation_closure is not None:
            # Call the closure with the name of the module and the input
            self.activation_closure(module._name, A)
        else:
            module._inputs = A

    def _process_grad(self, module, _, grad_out):
        """Process the incoming gradient wrt the output of the module."""

        G = grad_out[0]  # [N, S, O]

        if self.gradient_closure is not None:
            # Call the closure with the name of the module and the input
            self.gradient_closure(module._name, G)
        else:
            module._grads = G

    def __exit__(self, exc_type, exc, tb):
        # clean up secret attributes
        for layer in self.model.modules():
            if hasattr(layer, "_inputs"):
                del layer._inputs
            if hasattr(layer, "_name"):
                del layer._name

        # clean up hooks
        for h in self._fwd_hooks:
            h.remove()
        for h in self._bwd_hooks:
            h.remove()

        return False

    def covariances(self) -> Dict[str, TensorDict]:
        """Concatenate and flatten all the collected gradients into a single tensor."""
        grad_dict = TensorDict({})
        activation_dict = TensorDict({})
        for k, v in self.grad_covariance.items():
            grad_dict[k] = v
            activation_dict[k] = self.activation_covariance[k]

        return {"grad": grad_dict, "activation": activation_dict}


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # For multi-GPU
    np.random.seed(42)
    random.seed(42)
    # 1. Initialize distributed FIRST
    rank, world_size = setup_distributed()

    # 2. Initialize your checkpoint manager and task

    all_checkpoints = [[1000]]
    model_name = "EleutherAI/pythia-14m"

    pythia_checkpoints_manager = PythiaCheckpoints(all_checkpoints, model_name)
    pythia_checkpoints_manager.save_models(overwrite=False)

    assert pythia_checkpoints_manager.module_keys is not None

    task = LanguageModelingTask(module_keys=pythia_checkpoints_manager.module_keys)

    pythia_checkpoints_manager.module_keys = task.get_influence_tracked_modules()

    covariance_processor = CovarianceProcessor(task=task)

    # 3. Load dataset
    train_dataset = get_pile_dataset(model_str=model_name, step=0, max_samples=10_000)

    if rank == 0:
        print(f"Loaded {len(train_dataset)} samples from the Pile dataset.")  # type:ignore

    # 4. Load model with proper device handling
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=f"step{1000}",
        force_download=True,
        device_map=None,  # Don't use device_map with FSDP
    )

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though
    if rank == 0:
        print("-*" * 20)
        print(f"Loaded model {model_name} for training.")

    # 5. Apply FSDP2 wrapping BEFORE moving to device
    if dist.is_initialized() and world_size > 1:
        print("Applying FSDP2 wrapping to model layers...")
        print("-*" * 20)
        # Apply FSDP2 to transformer layers first
        for module in model.modules():
            if isinstance(module, GPTNeoXLayer):
                fully_shard(module)

        # Then wrap the entire model
        model = fully_shard(model)
        assert model is not None, "FSDP2 wrapping failed."
        model = model.to(device)  # Move to device after FSDP wrapping
        if rank == 0:
            print("-*" * 20)
            print("Applied FSDP2 sharding to model layers.")
    else:
        # Single GPU case - just move to device
        model = model.to(device)  # type: ignore
        if rank == 0:
            print("-*" * 20)
            print("Moved model to single GPU.")

    covariance_processor.compute_covariances(model=model, data=train_dataset)  # type: ignore
