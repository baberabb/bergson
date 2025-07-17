from contextlib import contextmanager
from typing import Generator

import torch
from numpy.lib.recfunctions import structured_to_unstructured
from torch import Tensor, nn

from .data import load_gradients
from .gradients import GradientCollector, GradientProcessor


class TraceResult:
    """Result of a .trace() call."""

    def __init__(self):
        # Should be set by the Attributor after a search
        self._indices: Tensor | None = None
        self._scores: Tensor | None = None

    @property
    def indices(self) -> Tensor:
        """The indices of the top-k examples."""
        if self._indices is None:
            raise ValueError("No indices available. Exit the context manager first.")

        return self._indices

    @property
    def scores(self) -> Tensor:
        """The attribution scores of the top-k examples."""
        if self._scores is None:
            raise ValueError("No scores available. Exit the context manager first.")

        return self._scores


class Attributor:
    def __init__(
        self,
        index_path: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
        unit_norm: bool = True,
    ):
        # Map the gradients into memory (very fast)
        mmap = load_gradients(index_path)
        if mmap.dtype.names is not None:
            mmap = structured_to_unstructured(mmap)

        # Load them onto the desired device (slow)
        self.grads = torch.tensor(mmap, device=device, dtype=dtype)

        # In-place normalize for numerical stability
        if unit_norm:
            self.grads /= self.grads.norm(dim=1, keepdim=True)

        # Load the gradient processor
        self.processor = GradientProcessor.load(index_path, map_location=device)

    def search(self, queries: Tensor, k: int) -> torch.return_types.topk:
        """
        Search for the `k` nearest examples in the index based on the query or queries.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        return torch.topk(queries @ self.grads.mT, k)

    @contextmanager
    def trace(
        self,
        module: nn.Module,
        k: int,
        *,
        precondition: bool = False,
    ) -> Generator[TraceResult, None, None]:
        """
        Context manager to trace the gradients of a module and return the
        corresponding Attributor instance.
        """
        mod_grads: list[Tensor] = []
        result = TraceResult()

        def callback(name: str, g: Tensor):
            # Precondition the gradient using Cholesky solve
            if precondition:
                P = self.processor.preconditioners[name]
                g = g.flatten(1).type_as(P)
                g = torch.cholesky_solve(g.mT, P).mT
            else:
                g = g.flatten(1)

            # Store the gradient for later use
            mod_grads.append(
                g.to(self.grads.device, self.grads.dtype, non_blocking=True)
            )

        with GradientCollector(module, callback, self.processor):
            yield result

        if not mod_grads:
            raise ValueError("No grads collected. Did you forget to call backward?")

        queries = torch.cat(mod_grads, dim=1)
        result._scores, result._indices = self.search(queries, k)
