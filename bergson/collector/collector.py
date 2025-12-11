import functools
import math
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import ContextDecorator, nullcontext
from dataclasses import astuple, dataclass, field
from typing import Callable, Literal, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, Value
from jaxtyping import Float
from torch import Tensor
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.hooks import RemovableHandle
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from bergson.collector.logger import get_logger
from bergson.config import AttentionConfig, IndexConfig, ReduceConfig
from bergson.data import Builder, pad_and_tensor
from bergson.gradients import GradientProcessor, LayerAdapter
from bergson.peft import set_peft_enabled
from bergson.score.scorer import Scorer
from bergson.utils import assert_type, create_projection_matrix


@dataclass
class HookCollectorBase(ContextDecorator, ABC):
    """
    Abstract base class for collectors that attach forward and backward hooks to model
    layers.

    Automatically discovers supported modules in the model, registers hooks during
    context entry, and provides lifecycle methods (setup/teardown) for subclasses to
    implement custom logic.

    Assumes module input shape is [N, S, I] where N=batch size, S=sequence length,
    I=input dimension.

    Subclasses must implement:
        - setup(): Initialize state (buffers, dicts, etc.)
        - teardown(): Clean up and save results
        - forward_hook(): Process activations during forward pass
        - backward_hook(): Process gradients during backward pass
    """

    model: nn.Module

    cfg: IndexConfig | None = None

    target_modules: set[str] | None = None
    """
    Set of module names to attach hooks to. Should consist only of supported modules.
    If None, hooks are attached to all Linear layers in the model.
    """

    processor: GradientProcessor = field(default_factory=GradientProcessor)
    """Configuration for processing and compressing gradients."""

    attention_cfgs: dict[str, AttentionConfig] = field(default_factory=dict)

    def __post_init__(
        self,
    ):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self._fwd_hooks: list[RemovableHandle] = []
        self._bwd_hooks: list[RemovableHandle] = []

        # Discover target Linear modules using the static method
        self.target_info = self.discover_targets(
            self.model, self.target_modules, self.processor.include_bias
        )

        # Allow subclasses to perform custom initialization
        self.setup()

    @staticmethod
    def discover_targets(
        model: nn.Module,
        target_modules: set[str] | None = None,
        include_bias: bool = False,
    ) -> dict[str, tuple[torch.device, torch.Size, bool]]:
        """
        Discover target Linear modules without instantiating a collector.

        This is useful when you need target_info early (e.g., to allocate buffers)
        before creating the actual collector instance.

        Args:
            model: The model to scan for Linear layers
            target_modules: Optional set of module names to filter. If None, all Linear
            layers are included.

        Returns:
            Dictionary mapping module names to (device, weight_shape) tuples.
        """
        target_info = {}
        for name, layer in model.named_modules():
            if not isinstance(layer, LayerAdapter.supported_modules):
                continue

            if target_modules is not None and name not in target_modules:
                continue

            has_bias = getattr(layer, "bias", None) is not None and include_bias

            target_info[name] = (
                layer.weight.device,
                layer.weight.shape,
                has_bias,
            )
        return target_info

    @staticmethod
    def get_head_name(name: str, head_idx: int) -> str:
        """Get the name of an attention head with index `head_idx` in a
        module with name `name`."""
        return f"{name}.head_{head_idx}"

    @staticmethod
    def split_attention_heads(fn):
        """Decorator that splits attention module calls into per-head calls."""

        @functools.wraps(fn)
        def wrapper(self, module, g):
            name = module._name

            if name not in self.attention_cfgs:
                return fn(self, module, g)

            num_heads, head_size, head_dim = astuple(self.attention_cfgs[name])

            # Save state
            orig_name = module._name
            orig_out = getattr(module, LayerAdapter.out_attr(module))

            setattr(module, LayerAdapter.out_attr(module), head_size)

            for h in range(num_heads):
                module._name = self.get_head_name(name, h)
                try:
                    head_g = torch.narrow(g, head_dim, h * head_size, head_size)
                except Exception as e:
                    print(
                        f"Error processing gradient of shape {g.shape} for head {h}"
                        f" in module {name}. Provided head config may be incorrect. "
                        f"Head config: head dim {head_dim}, head size {head_size},"
                        f" num heads {num_heads}."
                    )
                    raise e
                fn(self, module, head_g)

            # Restore
            module._name = orig_name
            setattr(module, LayerAdapter.out_attr(module), orig_out)

        return wrapper

    def shapes(self) -> Mapping[str, torch.Size]:
        """Return the shapes of the gradients collected by this collector."""
        proj_shape = (
            torch.Size((p_dim, p_dim))
            if (p_dim := self.processor.projection_dim) is not None
            else None
        )

        shapes = {}
        for name, (_, target_shape, has_bias) in self.target_info.items():
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
                    if has_bias:
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
                    if has_bias:
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
        key = (name, side, device)
        if key in self.processor._projection_matrices:
            return self.processor._projection_matrices[key]

        identifier = f"{name}/{side}"

        A = create_projection_matrix(
            identifier, m, n, dtype, device, self.processor.projection_type
        )
        self.processor._projection_matrices[key] = A
        return A

    def __enter__(self):
        """Register forward and backward hooks on all target modules."""
        for name in self.target_info:
            layer = self.model.get_submodule(name)

            # Store module name for use in hook callbacks
            layer._name = name  # type: ignore[attr-defined]
            layer._has_bias = self.target_info[name][2]  # type: ignore[attr-defined]

            # Register hooks
            fwd_hook = layer.register_forward_hook(self._process_input)
            self._fwd_hooks.append(fwd_hook)

            bwd_hook = layer.register_full_backward_hook(self._process_grad)
            self._bwd_hooks.append(bwd_hook)

        return self

    def _process_input(self, module: nn.Module, inp: tuple, _):
        """Internal forward hook that extracts input and delegates to subclass."""

        x = inp[0].detach()
        assert x.ndim == 3, f"Expected input of shape [N, S, I], got {x.shape}"

        self.forward_hook(module, x)

    def _process_grad(self, module: nn.Module, _, grad_out):
        """Internal backward hook that extracts gradient and delegates to subclass."""
        # Sanity checks
        assert isinstance(module, LayerAdapter.supported_modules), (
            f"Expected a module of type {LayerAdapter.supported_modules}, "
            f"got {type(module)}"
        )

        g = grad_out[0].detach()  # [N, S, O]

        self.backward_hook(module, g)

    def __exit__(self, exc_type, exc, tb):
        """Clean up hooks and allow subclass cleanup."""

        # Clean up temporary attributes
        for layer in self.model.modules():
            if hasattr(layer, "_inputs"):
                del layer._inputs
            if hasattr(layer, "_name"):
                del layer._name

        # Remove all registered hooks
        for h in self._fwd_hooks:
            h.remove()
        for h in self._bwd_hooks:
            h.remove()
        self._fwd_hooks.clear()
        self._bwd_hooks.clear()

        return False

    @abstractmethod
    def setup(self) -> None:
        """
        Called at the end of __post_init__.

        Override to perform custom initialization such as:
        - Allocating buffers or dictionaries
        - Loading pretrained weights or data
        - Initializing accumulators
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Called at the start of __exit__, before hooks are removed.

        Override to perform custom cleanup such as:
        - Saving results to disk
        - Flushing buffers
        - Computing final statistics
        - Freeing resources
        """
        pass

    @abstractmethod
    def forward_hook(self, module: nn.Module, a: Float[Tensor, "N S I"]) -> None:
        """
        Process activations during the forward pass.

        Args:
            name: Name of the module
            a: Input activations of shape [N, S, I]
        """
        pass

    @abstractmethod
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S I"]) -> None:
        """
        Process gradients during the backward pass.

        Args:
            name: Name of the module
            g: Gradient with respect to module output, shape [N, S, O]
        """
        pass

    @abstractmethod
    def process_batch(self, indices: list[int], **kwargs) -> None:
        """
        Process collected data for a batch.

        Args:
            indices: List of data indices in the current batch
            **kwargs: Additional batch-specific data (e.g., losses)
        """
        pass


@dataclass(kw_only=True)
class GradientCollector(HookCollectorBase):
    """
    Collects gradients for each Linear layer in the model.

    Stores gradients in a dictionary mapping module names to gradient tensors.
    """

    mod_grads: dict = field(default_factory=dict)

    data: Dataset

    reduce_cfg: ReduceConfig | None = None

    builder: Builder | None = None
    scorer: Scorer | None = None

    def setup(self) -> None:
        """Initialize gradient storage dictionary."""
        assert self.cfg is not None, "cfg is required for GradientCollector"
        assert isinstance(self.model.device, torch.device), (
            "Model device is not set correctly"
        )

        # TODO: handle more elegantly?
        self.save_dtype = (
            torch.float32 if self.model.dtype == torch.float32 else torch.float16
        )

        self.lo = torch.finfo(self.save_dtype).min
        self.hi = torch.finfo(self.save_dtype).max

        self.per_doc_losses = torch.full(
            (len(self.data),),
            device=self.model.device,
            dtype=self.save_dtype,
            fill_value=0.0,
        )

        # Compute whether we need to save the index
        self.save_index = self.scorer is None and not self.cfg.skip_index

        if self.save_index:
            grad_sizes = {name: math.prod(s) for name, s in self.shapes().items()}
            self.builder = Builder(
                self.cfg.partial_run_path,
                self.data,
                grad_sizes,
                self.save_dtype,
                self.reduce_cfg,
            )
        else:
            self.builder = None

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "N S I"]) -> None:
        p = self.processor.projection_dim
        name = assert_type(str, module._name)
        i = getattr(module, LayerAdapter.in_attr(module))

        if module._has_bias:
            # Append ones to activation for bias term
            ones = torch.ones(a.size(0), a.size(1), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=-1)
            i = i + 1
            setattr(module, LayerAdapter.in_attr(module), i)
        if p is not None:
            a = a @ self.projection(name, p, i, "right", a.device, a.dtype).T  # type: ignore

        module._inputs = a

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]):
        a = module._inputs  # [N, S, I/q]
        assert isinstance(a, torch.Tensor), "Activation cache missing for module"

        name = assert_type(str, module._name)

        p = self.processor.projection_dim
        o = getattr(module, LayerAdapter.out_attr(module))

        if p is not None:
            A = self.projection(name, p, o, "left", g.device, g.dtype)
            g = g @ A.T  # [N, S, p]

        P = g.mT @ a  # [N, O/p, S] @ [N, S, I/q] → [N, O/p, I/q]

        P = P.flatten(1).clamp_(self.lo, self.hi)

        if self.save_index:
            # Asynchronously move the gradient to CPU and convert to the final dtype
            self.mod_grads[name] = P.to(
                device="cpu", dtype=self.save_dtype, non_blocking=True
            )
        else:
            self.mod_grads[name] = P.to(dtype=self.save_dtype)

        del module._inputs

    def process_batch(self, indices: list[int], **kwargs):
        """Process collected gradients for a batch and update losses."""
        losses = kwargs.get("losses")
        assert losses is not None, "losses must be provided in kwargs"

        if self.builder:
            self.builder(indices, self.mod_grads)
        if self.scorer:
            self.scorer(indices, self.mod_grads)
        self.mod_grads.clear()
        self.per_doc_losses[indices] = losses.detach().type_as(self.per_doc_losses)

    def teardown(self):
        assert self.cfg is not None, (
            "cfg is required for GradientCollector"
        )  # pleasing type checker
        if dist.is_initialized():
            dist.reduce(self.per_doc_losses, dst=0)

        if self.rank == 0:
            if self.cfg.drop_columns:
                data = self.data.remove_columns(["input_ids"])

            data = self.data.add_column(
                "loss",
                self.per_doc_losses.cpu().numpy(),
                feature=Value(
                    "float16"
                    if self.save_dtype == torch.float16
                    else "float32"  # TODO: This is not robust
                ),
                new_fingerprint="loss",
            )

            data.save_to_disk(self.cfg.partial_run_path / "data.hf")

            self.processor.save(self.cfg.partial_run_path)

        # Flush and reduce builder if it exists
        if self.builder is not None:
            self.builder.flush()
            self.builder.dist_reduce()


@dataclass(kw_only=True)
class TraceCollector(HookCollectorBase):
    """
    Collects gradients for each Linear layer in the model.

    Stores gradients in a dictionary mapping module names to gradient tensors.
    """

    mod_grads: dict = field(default_factory=lambda: defaultdict(list))

    precondition: bool = False

    device: torch.device | str

    dtype: torch.dtype

    def setup(self) -> None:
        return

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "N S I"]) -> None:
        p = self.processor.projection_dim
        name = assert_type(str, module._name)
        i = getattr(module, LayerAdapter.in_attr(module))

        if module._has_bias:
            # Append ones to activation for bias term
            ones = torch.ones(a.size(0), a.size(1), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=-1)
            i = i + 1
            setattr(module, LayerAdapter.in_attr(module), i)
        if p is not None:
            a = a @ self.projection(name, p, i, "right", a.device, a.dtype).T  # type: ignore

        module._inputs = a

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]):
        a = module._inputs  # [N, S, I/q]
        assert isinstance(a, torch.Tensor), "Activation cache missing for module"

        name = assert_type(str, module._name)

        p = self.processor.projection_dim

        o = getattr(module, LayerAdapter.out_attr(module))

        if p is not None:
            A = self.projection(name, p, o, "left", g.device, g.dtype)
            g = g @ A.T  # [N, S, p]

        P = g.mT @ a  # [N, O/p, S] @ [N, S, I/q] → [N, O/p, I/q]

        P = P.flatten(1)

        # Precondition the gradient using Cholesky solve
        # TODO: Should damp here?
        if self.precondition:
            eigval, eigvec = self.processor.preconditioners_eigen[name]
            eigval_inverse_sqrt = 1.0 / (eigval).sqrt()
            prec = eigvec * eigval_inverse_sqrt @ eigvec.mT
            P = P.type_as(prec) @ prec  # <- apply to P

        # Store the gradient for later use
        self.mod_grads[name].append(P.to(self.device, self.dtype, non_blocking=True))

    def process_batch(self, indices: list[int], **kwargs):
        return

    def teardown(self):
        return


@dataclass
class StreamingGradientCollector(HookCollectorBase):
    """
    Lightweight collector for streaming gradient collection during training.

    Stores per-sample gradients in `mod_grads` dict for external consumers
    (e.g., callbacks) to process. Does not manage file I/O internally.
    """

    mod_grads: dict = field(default_factory=dict)

    save_dtype: torch.dtype = torch.float16

    def setup(self) -> None:
        self.lo = torch.finfo(self.save_dtype).min
        self.hi = torch.finfo(self.save_dtype).max

    def teardown(self) -> None:
        pass

    def process_batch(self, indices: list[int], **kwargs) -> None:
        pass

    def forward_hook(self, module: nn.Module, a: Float[Tensor, "N S I"]) -> None:
        p = self.processor.projection_dim
        name = assert_type(str, module._name)
        i = getattr(module, LayerAdapter.in_attr(module))

        if module._has_bias:
            ones = torch.ones(a.size(0), a.size(1), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=-1)
            i = i + 1
            setattr(module, LayerAdapter.in_attr(module), i)

        if p is not None:
            a = a @ self.projection(name, p, i, "right", a.device, a.dtype).T

        module._inputs = a

    @HookCollectorBase.split_attention_heads
    def backward_hook(self, module: nn.Module, g: Float[Tensor, "N S O"]) -> None:
        a = module._inputs
        assert isinstance(a, torch.Tensor), "Activation cache missing for module"

        name = assert_type(str, module._name)

        p = self.processor.projection_dim
        o = getattr(module, LayerAdapter.out_attr(module))

        if p is not None:
            A = self.projection(name, p, o, "left", g.device, g.dtype)
            g = g @ A.T

        P = g.mT @ a
        P = P.flatten(1).clamp_(self.lo, self.hi)

        self.mod_grads[name] = P.to(
            device="cpu", dtype=self.save_dtype, non_blocking=True
        )

        del module._inputs


class CollectorComputer:
    """Generic Computer that computes collectors."""

    def __init__(
        self,
        model: PreTrainedModel,
        data: Dataset,
        *,
        collector: HookCollectorBase,
        batches: list[list[int]] | None = None,
        cfg: IndexConfig,
        **kwargs,
    ):
        # Model
        self.model = model
        self.device = model.device

        # Data
        self.data = data
        # Batch size one by default
        if batches is None:
            batches = [[idx] for idx in range(len(data))]
        self.batches = batches

        self.loss_fn = loss_fn_factory(cfg)

        # Collector
        self.collector = collector

        # Other
        self.cfg = cfg
        self.logger = get_logger(
            "CollectorComputer", level="DEBUG" if cfg.debug else "INFO"
        )

        # Distributed related
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.logger.info("Computing with collector for target modules.")

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

    def _compute(
        self,
        desc: Optional[str] = None,
    ):
        total_processed = torch.tensor(0, device=self.model.device)
        prof = self._setup_profiler()
        step = 0

        with prof:
            for indices in tqdm(
                self.batches, disable=self.rank != 0, desc=f"Computing {desc}"
            ):
                batch = self.data[indices]

                with (
                    self.collector,
                    (
                        record_function(f"step_{step}")
                        if self.cfg.profile
                        else nullcontext()
                    ),
                ):
                    losses = self.loss_fn(self.model, batch)
                    losses.mean().backward()

                    self.model.zero_grad()
                    # TODO: currently builder also calls torch.cuda.synchronize
                    torch.cuda.synchronize() if torch.cuda.is_available() else None

                if self.cfg.profile:
                    assert isinstance(prof, profile), "Profiler is not set up correctly"
                    prof.step()
                step += 1

                self.collector.process_batch(indices, losses=losses)

        self.collector.teardown()
        if dist.is_initialized():
            dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)
        self.logger.info(f"Total processed: {total_processed.item()}")


def loss_fn_factory(cfg: IndexConfig) -> Callable:
    """Factory to create loss functions based on type."""

    def loss_fn(model, batch):
        x, y = pad_and_tensor(
            batch["input_ids"],  # type: ignore
            labels=batch.get("labels"),  # type: ignore
            device=model.device,
        )
        logits = model(x).logits[:, :-1]
        masks = y[:, 1:] != -100
        denoms = (
            masks.sum(dim=1, dtype=model.dtype) if cfg.loss_reduction == "mean" else 1.0
        )

        if cfg.loss_fn == "kl":
            with torch.inference_mode():
                set_peft_enabled(model, False)
                ref_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)
                set_peft_enabled(model, True)

            ft_lps = torch.log_softmax(logits, dim=-1)

            # Compute average KL across all unmasked tokens
            kls = torch.sum(ft_lps.exp() * (ft_lps - ref_lps), dim=-1)
            losses = torch.sum(kls * masks, dim=-1) / denoms
            if "advantage" in batch:
                losses *= torch.tensor(batch["advantage"], device=losses.device)

        else:
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y[:, 1:].flatten(),
                reduction="none",
            ).reshape_as(y[:, 1:])
            losses = losses.sum(1) / denoms
            if "advantage" in batch:
                losses *= torch.tensor(batch["advantage"], device=losses.device)

        return losses

    return loss_fn
