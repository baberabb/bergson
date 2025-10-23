import json
import os
from abc import ABC, abstractmethod
from contextlib import ContextDecorator
from dataclasses import asdict, astuple, dataclass, field
from typing import Callable, Literal, Mapping

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from transformers.pytorch_utils import Conv1D as HFConv1D

from .data import AttentionConfig
from .math import reshape_to_nearest_square
from .utils import assert_type, create_projection_matrix

NORMALIZER_TYPES: dict[str, type["Normalizer"]] = {}


class Normalizer(ABC):
    """
    Base class for normalizers that can be used to scale gradients.
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in the NORMALIZER_TYPES dict."""
        super().__init_subclass__(**kwargs)
        NORMALIZER_TYPES[cls.__name__] = cls

    @staticmethod
    def from_state_dict(state_dict: dict[str, str | Tensor]) -> "Normalizer":
        """
        Create a normalizer instance from a state dictionary.
        The state dictionary should contain the class name and the tensors.
        """
        class_name = state_dict.pop("__class__")
        assert isinstance(class_name, str), "Expected '__class__' to be a string"

        if (cls := NORMALIZER_TYPES.get(class_name)) is None:
            raise ValueError(f"Unknown normalizer class: '{class_name}'")

        return cls(**state_dict)

    @abstractmethod
    def normalize_(
        self,
        grad: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Normalize gradients in-place, adding a small epsilon to avoid division by zero.
        """

    def state_dict(self) -> dict[str, str | Tensor]:
        """
        Return the state of the normalizer as a dictionary of tensors.
        This is used for saving and loading the normalizer.
        """
        tensors = {k: v for k, v in self.__dict__.items() if isinstance(v, Tensor)}
        return {
            "__class__": self.__class__.__name__,
            **tensors,
        }


@dataclass
class AdafactorNormalizer(Normalizer):
    """
    Row and column sums of second moments of gradients for a matrix-valued parameter.
    """

    row: Tensor  # shape [O]
    col: Tensor  # shape [I]

    def __post_init__(self):
        assert self.row.ndim == 1, f"Expected 1D tensor for row, got {self.row.ndim}D"
        assert self.col.ndim == 1, f"Expected 1D tensor for col, got {self.col.ndim}D"

    @torch.compile
    def normalize_(
        self,
        grad: Tensor,
        eps: float = 1e-30,
    ) -> Tensor:
        """
        Normalize the row and column sums by adding a small epsilon.

        Note: Our `eps` corresponds to epsilon_1 in the original Adafactor paper. They
        recommend 1e-30, but we use 1e-16 for extra numerical stability.
        """
        # We follow the Adafactor implementation in the tensor2tensor repo, which is
        # different from the paper and from the PyTorch implementation. First add eps
        # to ensure these second moments are sufficiently far from zero. Then we don't
        # need to worry about numerical stability anywhere else, and we don't need to
        # materialize the outer product at any point.
        r, c = self.row.add(eps), self.col.add(eps)

        # This is the denominator for V, the rank-one matrix of second moment estimates:
        # V = torch.outer(r, c) / denom
        # V_ij = r_i * c_j / denom
        # But we want to (implicitly) take the Hadamard product with the elementwise
        # reciprocal square root of V:
        # (V_ij)^{-1/2} = denom.sqrt() * r_i.rsqrt() * c_j.rsqrt()
        denom = r.mean()

        # Hadamard product with a rank-one matrix ab^T is the same as left-multiplying
        # by diag(a) and right-multiplying by diag(b). In this case we can represent
        # the elementwise reciprocal square root of V as ab^T where:
        # a = denom.sqrt() * r.rsqrt() and b = c.rsqrt()
        a = denom.sqrt() * r.rsqrt_()  # shape [O]
        b = c.rsqrt_()

        # Implicitly do the Hadamard product
        grad *= a[:, None]  # [N, O] * [O] → [N, O]
        grad *= b[None, :]
        return grad

    def to_adam(self) -> "AdamNormalizer":
        """
        Convert this Adafactor normalizer to an Adam normalizer by materializing the
        rank-one second moment matrix.
        """
        # Compute the second moment matrix as a square matrix of shape [O, I]
        # NOTE: We don't add the epsilon here, since the AdamNormalizer is going to
        # add it outside the square root. This could cause infs though if there are
        # any exactly zero rows or columns, so we should be careful.
        avg_sq = torch.outer(self.row, self.col) / self.row.mean()
        return AdamNormalizer(avg_sq=avg_sq)


@dataclass
class AdamNormalizer(Normalizer):
    """
    Contains the second moments of the gradients.
    """

    avg_sq: Tensor

    @torch.compile
    def normalize_(
        self,
        grad: Tensor,
        eps: float = 1e-8,
    ) -> Tensor:
        """Normalize the gradients by the square root of the second moments."""
        # Adam-style epsilon is added outside the square root
        denom = self.avg_sq.sqrt()
        return grad.div_(denom.add_(eps))

    def to_adafactor(self) -> AdafactorNormalizer:
        """
        Convert this Adam normalizer to an Adafactor normalizer, minimizing the
        I-divergence (generalized Kullback-Leibler divergence) between the original
        and the factored second moments.
        """
        # We assume avg_sq is a square matrix of shape [O, I]
        assert (
            self.avg_sq.ndim == 2
        ), f"Expected 2D tensor for avg_sq, got {self.avg_sq.ndim}D"

        # Compute row and column means
        return AdafactorNormalizer(
            row=self.avg_sq.mean(dim=1),  # shape [O]
            col=self.avg_sq.mean(dim=0),  # shape [I]
        )


@dataclass
class GradientProcessor:
    """Configuration for processing and compressing gradients."""

    normalizers: Mapping[str, Normalizer] = field(default_factory=dict)
    """
    Dictionary of normalizers for each matrix-valued parameter in the model. The keys
    should match the names of the parameters in the model. If a parameter does not have
    a normalizer, it will be skipped.
    """

    preconditioners: dict[str, Tensor] = field(default_factory=dict)
    """
    Dictionary of preconditioners for each matrix-valued parameter in the model.
    These are applied after the normalization and random projection steps.
    """

    preconditioners_eigen: Mapping[str, tuple[Tensor, Tensor]] = field(
        default_factory=dict
    )
    """
    Dictionary of eigen decompositions of preconditioners for each matrix-valued
    parameter in the model. Each value is a tuple of (eigenvalues, eigenvectors).
    These are used to efficiently apply inverse square-root of the preconditioners
    to the gradients."""

    projection_dim: int | None = None
    """Number of rows and columns to project the gradients to. If `None`, keep the
    original shape of the gradients."""

    reshape_to_square: bool = False
    """Whether to reshape the gradients into a nearly square matrix before projection.
    This is useful when the matrix-valued parameters are far from square, like in the
    case of LoRA adapters."""

    projection_type: Literal["normal", "rademacher"] = "rademacher"
    """
    Type of random projection to use for compressing gradients. Can be either "normal"
    for Gaussian projections or "rademacher" for Rademacher projections, which use a
    uniform distribution over {-1, 1}.
    """

    include_bias: bool = False
    """Whether to include bias gradients when present on a module."""

    def __post_init__(self):
        self._projection_matrices: dict[
            tuple[str, Literal["left", "right"], torch.device, int, int], Tensor
        ] = {}

    @classmethod
    def load(
        cls,
        path: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "GradientProcessor":
        """
        Load the normalizers and preconditioners from a file.
        """
        cfg_path = os.path.join(path, "processor_config.json")
        norm_path = os.path.join(path, "normalizers.pth")
        precond_path = os.path.join(path, "preconditioners.pth")
        precond_eigen_path = os.path.join(path, "preconditioners_eigen.pth")

        # Load configuration
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Backward compatibility
        if "projection_type" not in cfg:
            cfg["projection_type"] = "normal"
        if "include_bias" not in cfg:
            cfg["include_bias"] = False

        # Load normalizers
        norm_state = torch.load(
            norm_path,
            map_location=map_location,
            weights_only=True,
        )
        normalizers = {
            name: Normalizer.from_state_dict(state)
            for name, state in norm_state.items()
        }

        return cls(
            normalizers=normalizers,
            preconditioners=torch.load(
                precond_path,
                map_location=map_location,
                weights_only=True,
            ),
            preconditioners_eigen=torch.load(
                precond_eigen_path,
                map_location=map_location,
                weights_only=True,
            ),
            **cfg,
        )

    def save(self, path: str):
        """
        Save the normalizers and preconditioners to a file.
        """
        os.makedirs(path, exist_ok=True)

        cfg_path = os.path.join(path, "processor_config.json")
        norm_path = os.path.join(path, "normalizers.pth")
        precond_path = os.path.join(path, "preconditioners.pth")
        precond_eigen_path = os.path.join(path, "preconditioners_eigen.pth")

        # Save configuration separately
        cfg = asdict(self)
        del cfg["normalizers"]
        del cfg["preconditioners"]
        del cfg["preconditioners_eigen"]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Save normalizers
        norm_state = {
            name: normalizer.state_dict()
            for name, normalizer in self.normalizers.items()
        }
        torch.save(norm_state, norm_path)
        torch.save(self.preconditioners, precond_path)
        torch.save(self.preconditioners_eigen, precond_eigen_path)


class LayerAdapter:
    supported_modules = (nn.Linear, HFConv1D, nn.Conv1d, nn.Conv2d, nn.Conv3d)

    @staticmethod
    def in_attr(layer: nn.Module) -> str:
        match layer:
            case nn.Linear():
                return "in_features"
            case HFConv1D():
                return "nx"
            case nn.Conv1d() | nn.Conv2d() | nn.Conv3d():
                return "in_channels"
            case _:
                raise ValueError(f"Unsupported layer type: {type(layer)}")

    @staticmethod
    def out_attr(layer: nn.Module) -> str:
        match layer:
            case nn.Linear():
                return "out_features"
            case HFConv1D():
                return "nf"
            case nn.Conv1d() | nn.Conv2d() | nn.Conv3d():
                return "out_channels"
            case _:
                raise ValueError(f"Unsupported layer type: {type(layer)}")


@dataclass
class GradientCollector(ContextDecorator):
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
    matrices in modules supported by LayerAdapter. If `None`, the gradients for all
    weight matrices will be collected.
    """

    attention_cfgs: dict[str, AttentionConfig] = field(default_factory=dict)
    """
    Dictionary of head configurations for each module to be split into head matrices.
    """

    def __post_init__(self):
        self._fwd_hooks: list[RemovableHandle] = []
        self._bwd_hooks: list[RemovableHandle] = []

        self.target_info: dict[str, tuple[torch.device, torch.Size, bool]] = {}

        # Before we add any hooks, we need to peek at what modules we need to track.
        for name, layer in self.model.named_modules():
            if not isinstance(layer, LayerAdapter.supported_modules):
                continue

            if self.target_modules is not None and name not in self.target_modules:
                continue

            # Users of this class really like to know ahead of time what the shapes are
            has_bias = getattr(layer, "bias", None) is not None
            self.target_info[name] = (
                layer.weight.device,
                layer.weight.shape,
                has_bias,
            )

    def shapes(self) -> Mapping[str, torch.Size]:
        """Return the shapes of the gradients collected by this collector."""
        proj_shape = (
            torch.Size((p_dim, p_dim))
            if (p_dim := self.processor.projection_dim) is not None
            else None
        )

        shapes = {}
        for name, (_, target_shape, has_bias) in self.target_info.items():
            include_bias = has_bias and self.processor.include_bias
            if name in self.attention_cfgs:
                attention_cfg = self.attention_cfgs[name]
                if proj_shape:
                    head_shape = proj_shape
                else:
                    # Mutate the attention module's shape to get the attention
                    # head shape
                    attention_shape = list(target_shape)
                    # - 2 because we're excluding the batch and sequence activation
                    # dimensions
                    attention_shape[attention_cfg.head_dim - 2] = (
                        attention_cfg.head_size
                    )
                    if include_bias:
                        attention_shape[-1] += 1
                    head_shape = torch.Size(attention_shape)

                shapes.update(
                    {
                        self.get_head_name(name, h): head_shape
                        for h in range(attention_cfg.num_heads)
                    }
                )
            else:
                if proj_shape:
                    shapes[name] = proj_shape
                else:
                    grad_shape = list(target_shape)
                    if include_bias:
                        grad_shape[-1] += 1
                    shapes[name] = torch.Size(grad_shape)

        return shapes

    def projection(
        self,
        name: str,
        m: int,
        n: int,
        side: Literal["left", "right"],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the `side` projection matrix for parameter `name` of shape [m, n]."""
        key = (name, side, device, m, n)
        if key in self.processor._projection_matrices:
            return self.processor._projection_matrices[key]

        identifier = f"{name}/{side}"

        A = create_projection_matrix(
            identifier, m, n, dtype, device, self.processor.projection_type
        )
        self.processor._projection_matrices[key] = A
        return A

    def get_head_name(self, name: str, head_idx: int) -> str:
        """Get the name of an attention head with index `head_idx` in a
        module with name `name`."""
        return f"{name}.head_{head_idx}"

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

        return self

    def _save_input(self, module: nn.Module, inp: tuple, _):
        """Save the input to the module for later use in the backward pass."""
        x = inp[0].detach()
        assert x.ndim == 3, f"Expected input of shape [N, S, I], got {x.shape}"

        # Pre-scale the input by the Adafactor column stats
        name = assert_type(str, module._name)
        norm = self.processor.normalizers.get(name)
        if isinstance(norm, AdafactorNormalizer):
            b = norm.col.add(1e-30)
            b.rsqrt_()

            x = x * b.type_as(x)  # [N, S, I] * [I] → [N, S, I]

        include_bias = (
            getattr(module, "bias", None) is not None and self.processor.include_bias
        )

        # If we're not using AdamNormalizer, we can randomly project the input here
        # to save memory, rather than waiting until the backward pass.
        p = self.processor.projection_dim
        if p is not None and not isinstance(norm, AdamNormalizer) and not include_bias:
            i = getattr(module, LayerAdapter.in_attr(module))
            x = x @ self.projection(name, p, i, "right", x.device, x.dtype).T  # type: ignore

        module._inputs = x

    def _process_grad(self, module: nn.Module, _, grad_out):
        """Process the incoming gradient wrt the output of the module."""
        # Sanity checks
        assert isinstance(module, LayerAdapter.supported_modules), (
            f"Expected a module of type {LayerAdapter.supported_modules}, "
            f"got {type(module)}"
        )
        G = grad_out[0]  # [N, S, O]
        I = module._inputs  # [N, S, I/q]

        name = assert_type(str, module._name)
        module_has_bias = getattr(module, "bias", None) is not None
        include_bias = module_has_bias and self.processor.include_bias

        if name in self.attention_cfgs:
            # Recurse into heads with module mutation and restoration
            num_heads, head_size, head_dim = astuple(self.attention_cfgs[name])

            module_name, module_inputs, module_out_features = (
                module._name,
                module._inputs,
                getattr(module, LayerAdapter.out_attr(module)),
            )
            setattr(module, LayerAdapter.out_attr(module), head_size)
            for h in range(num_heads):
                module._name = self.get_head_name(name, h)  # type: ignore
                module._inputs = module_inputs

                try:
                    head_G = torch.narrow(G, head_dim, h * head_size, head_size)
                except Exception as e:
                    print(
                        f"Error processing gradient of shape {G.shape} for head {h}"
                        f" in module {name}. Provided head config may be incorrect. "
                        f"Head config: head dim {head_dim}, head size {head_size},"
                        f" num heads {num_heads}."
                    )
                    raise e

                self._process_grad(module, None, (head_G,))
            module._name, module._inputs = (module_name, module_inputs)
            setattr(module, LayerAdapter.out_attr(module), module_out_features)

            return

        p = self.processor.projection_dim
        i = getattr(module, LayerAdapter.in_attr(module))
        o = getattr(module, LayerAdapter.out_attr(module))

        # Pre-scale G by the Adafactor row statistics
        norm = self.processor.normalizers.get(name)
        if isinstance(norm, AdafactorNormalizer):
            # Compare to the normalize_ method in AdafactorNormalizer
            r = norm.row.add(1e-30)
            a = r.mean().sqrt() * r.rsqrt_()

            G = G * a.type_as(G)  # [N, S, O] * [O] → [N, S, O]

        # For Adam, we need to materialize the full gradient and then project
        if isinstance(norm, AdamNormalizer) or include_bias:

            P = G.mT @ I  # [N, O, S] @ [N, S, I] → [N, O, I]
            if include_bias:
                # Append the bias gradient to the input
                P = torch.cat(
                    [
                        P,
                        G.sum(dim=(0, 1))
                        .unsqueeze(0)
                        .unsqueeze(2)
                        .expand(P.shape[0], -1, 1),
                    ],
                    dim=2,
                )
                i += 1

            if isinstance(norm, AdamNormalizer):
                # Normalize the gradients using the second moment matrix
                P /= norm.avg_sq.sqrt().add_(1e-8)

            if self.processor.reshape_to_square:
                P = reshape_to_nearest_square(P)
                o, i = P.shape[-2:]

            # Project the gradients to the lower-dimensional space
            if p is not None:
                A = self.projection(name, p, o, "left", P.device, G.dtype)
                B = self.projection(name, p, i, "right", P.device, G.dtype)
                P = A @ P @ B.T  # [N, p, q]

        # Both Adafactor and no normalizer, we can project G first
        else:
            if p is not None:
                A = self.projection(name, p, o, "left", G.device, G.dtype)
                G = G @ A.T  # [N, S, p]

            P = G.mT @ I  # [N, O/p, S] @ [N, S, I/q] → [N, O/p, I/q]

        self.closure(name, P)

        # Save memory ASAP
        del module._inputs

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
