import json
import os
import random
from contextlib import ContextDecorator
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Mapping

import debugpy
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import send_to_device
from datasets import Dataset
from safetensors.torch import save_file
from torch import Tensor, autocast
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm
from transformers import default_data_collator

from bergson.approx_unrolling.language_task import LanguageModelingTask, Task
from bergson.approx_unrolling.model_checkpoints import PythiaCheckpoints
from bergson.approx_unrolling.pile_data import get_pile_dataset
from bergson.approx_unrolling.utils import TensorDict
from bergson.gradients import Normalizer

NORMALIZER_TYPES: dict[str, type["Normalizer"]] = {}

NORMALIZER_TYPES: dict[str, type["Normalizer"]] = {}

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)


@dataclass
class CovarianceProcessor:
    """Configuration for processing and compressing gradients."""

    task: Task
    """
    The task in question, in particular defines the training loss function
    """

    gradient_covariance: Mapping[str, Tensor] = field(default_factory=dict)
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
        data: Dataset,
    ):
        """
        Estimate preconditioners from data. Overwrites the `preconditioners` field.
        """

        rank = dist.get_rank() if dist.is_initialized() else 0
        model.to(device)

        dataloader = DataLoader(
            data,
            batch_size=32,
            sampler=SequentialSampler(data_source=data),
            collate_fn=default_data_collator,
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
            batch = send_to_device(batch, device)

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
                activation_covariance /= dist.get_world_size()

            for gradient_covariance in mgr.grad_covariance.values():
                dist.all_reduce(gradient_covariance)
                gradient_covariance /= dist.get_world_size()

        # save using safetensors
        save_dir = "influence_results_closure"
        os.makedirs(save_dir, exist_ok=True)
        save_file(activation_covariances, os.path.join(save_dir, "activation_covariance.safetensors"))
        save_file(gradient_covariances, os.path.join(save_dir, "gradient_covariance.safetensors"))
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

        return cls(
            normalizers=normalizers,
            gradient_covariance=torch.load(
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

        # Save configuration separately
        cfg = asdict(self)
        del cfg["normalizers"]
        del cfg["preconditioners"]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Save normalizers
        norm_state = {name: normalizer.state_dict() for name, normalizer in self.normalizers.items()}
        torch.save(norm_state, norm_path)
        torch.save(self.preconditioners, precond_path)


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

    processor: CovarianceProcessor = field(default_factory=CovarianceProcessor)
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
        if module._name == "gpt_neox.layers.0.attention.dense":
            debugpy.breakpoint()
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
    all_checkpoints = [[1000]]
    model_name = "EleutherAI/pythia-14m"

    pythia_checkpoints_manager = PythiaCheckpoints(all_checkpoints, model_name)
    pythia_checkpoints_manager.save_models(overwrite=False)

    assert pythia_checkpoints_manager.module_keys is not None

    task = LanguageModelingTask(module_keys=pythia_checkpoints_manager.module_keys)

    pythia_checkpoints_manager.module_keys = task.get_influence_tracked_modules()

    model = pythia_checkpoints_manager.load_checkpoint(checkpoint=1000)

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    covariance_processor = CovarianceProcessor(task=task)
    train_dataset = get_pile_dataset(model_str=model_name, step=0, max_samples=1000)
    # covariance_processor.compute_covariances(model=model, data=train_dataset)

    # torch.cuda.memory._record_memory_history(max_entries=100000)

    covariance_processor.compute_covariances(model=model, data=train_dataset)

    # torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    # torch.cuda.memory._record_memory_history(enabled=None)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # ) as prof:
    #     covariance_processor.compute_covariances(model=model, data=train_dataset)
    # prof.export_chrome_trace("profile_sample_true_gc.json")
