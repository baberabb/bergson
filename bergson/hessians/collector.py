import hashlib
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from bergson.gradients import GradientProcessor
from bergson.utils import assert_type


@dataclass
class EkfacCollector(ContextDecorator):
    """
    Adds forward and backward hooks to `model` that efficiently collect per-sequence
    gradients for all the matrix-valued parameters, randomly projecting them using a
    fixed seed to compress them into lower-dimensional blocks of shape [p×q]. We use
    a dictionary of `AdafactorNormalizer` to scale the gradients by the second moments
    of the parameters, which are expected to be precomputed and passed in.

    We assume that the input to `model` is of shape `[N, S, I]`, where `N` is the
    batch size, `S` is the sequence length, and `I` is the input dimension. We take the
    mean over the sequence length to obtain a single gradient per sequence.
    """

    model: nn.Module

    closure: Callable
    """Closure to call on the gradient as it is collected."""

    processor: GradientProcessor = field(default_factory=GradientProcessor)
    """Configuration for processing and compressing gradients."""

    target_modules: set[str] | None = None
    """
    List of parameter names to collect gradients for. Should consist only of weight
    matrices in `nn.Linear` modules. If `None`, the gradients for all weight matrices
    will be collected.
    """

    fwd_closure: Optional[Callable] = None
    """Closure to call on the activations during forward hook."""

    def __post_init__(self):
        self._fwd_hooks: list[RemovableHandle] = []
        self._bwd_hooks: list[RemovableHandle] = []

        self.target_info: dict[str, tuple[torch.device, torch.Size]] = {}

        # Before we add any hooks, we need to peek at what modules we need to track.
        for name, layer in self.model.named_modules():
            if not isinstance(layer, nn.Linear):
                continue

            if self.target_modules is not None and name not in self.target_modules:
                continue

            if "attn" in name:
                continue

            # Users of this class really like to know ahead of time what the shapes are
            self.target_info[name] = layer.weight.device, layer.weight.shape

    def shapes(self) -> Mapping[str, torch.Size]:
        """Return the shapes of the gradients collected by this collector."""
        if (p_dim := self.processor.projection_dim) is not None:
            return {name: torch.Size((p_dim, p_dim)) for name in self.target_info}

        # If we don't have a projection dimension, we can just use the original shapes.
        return {name: shape for name, (_, shape) in self.target_info.items()}

    def projection(
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

        device, _ = self.target_info[name]
        prng = torch.Generator(device).manual_seed(seed)

        A = torch.randn(m, n, device=device, dtype=dtype, generator=prng)
        A /= A.norm(dim=1, keepdim=True)
        return A

    def __enter__(self):
        # Install a hook on every Linear
        for name in self.target_info:
            layer = self.model.get_submodule(name)

            # Save the name of the layer for later use
            layer._name = name  # type: ignore[attr-defined]

            # register forward hook to save V = X @ B^T
            fwd_hook = layer.register_forward_hook(self._save_input)
            self._fwd_hooks.append(fwd_hook)

            # register backward hook to compute P = sum(U @ V^T)
            bwd_hook = layer.register_full_backward_hook(self._process_grad)
            self._bwd_hooks.append(bwd_hook)
            # self._register_post_forward_cleanup()

        return self

    # def _register_post_forward_cleanup(self):
    #     """Register a hook that cleans up GPU cache after forward pass completes."""
    #     # Find the last layer we're hooking
    #     last_layer_name = max(self.target_info.keys())  # or however you determine order
    #     last_layer = self.model.get_submodule(last_layer_name)
    #     print(f"---- {last_layer_name}")

    #     def cleanup_cache(module, inp, out):
    #         torch.cuda.empty_cache()
    #         # Optionally log memory usage
    #         print(f"GPU memory after cache clear: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    #     cleanup_hook = last_layer.register_forward_hook(cleanup_cache)
    #     self._fwd_hooks.append(cleanup_hook)

    def _save_input(self, module: nn.Module, inp: tuple, _):
        name = assert_type(str, module._name)

        """Save the input to the module for later use in the backward pass."""
        x = inp[0].detach()
        assert x.ndim == 3, f"Expected input of shape [N, S, I], got {x.shape}"

        if self.fwd_closure:
            self.fwd_closure(name, x)
        else:
            module._inputs = x  # type: ignore[attr-defined]

    def _process_grad(self, module: nn.Module, _, grad_out):
        """Process the incoming gradient wrt the output of the module."""
        # Sanity checks
        assert isinstance(module, nn.Linear), "Expected a Linear module"
        G = grad_out[0]  # [N, S, O]

        name = assert_type(str, module._name)
        if self.fwd_closure is None:
            assert hasattr(module, "_inputs"), "Expected inputs to be saved in the module"
            X = module._inputs
            G = G.mT @ X
        if self.closure:
            self.closure(name, G)
            return

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
