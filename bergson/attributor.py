from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import torch
from torch import Tensor, nn

from .data import load_gradients
from .faiss_index import FaissConfig, FaissIndex
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
        index_path: Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        unit_norm: bool = False,
        faiss_cfg: FaissConfig | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.unit_norm = unit_norm
        self.faiss_index = None

        # Load the gradient processor
        self.processor = GradientProcessor.load(index_path, map_location=device)

        # Load the gradients into a FAISS index
        if faiss_cfg:
            faiss_index_name = (
                f"faiss_{faiss_cfg.index_factory.replace(',', '_')}"
                f"{'_cosine' if unit_norm else ''}"
            )
            faiss_path = index_path / faiss_index_name

            if not (faiss_path / "config.json").exists():
                FaissIndex.create_index(
                    index_path, faiss_path, faiss_cfg, device, unit_norm
                )

            self.faiss_index = FaissIndex(
                faiss_path, device, mmap_index=faiss_cfg.mmap_index
            )
            self.N = self.faiss_index.ntotal
            self.ordered_modules = self.faiss_index.ordered_modules
            return

        # Load the gradients into memory
        mmap = load_gradients(index_path)

        # Copy gradients into device memory
        self.grads = {
            name: torch.tensor(mmap[name], device=device, dtype=dtype)
            for name in mmap.dtype.names
        }
        self.N = mmap[mmap.dtype.names[0]].shape[0]

        self.ordered_modules = mmap.dtype.names

        if unit_norm:
            norm = torch.cat(
                [self.grads[name] for name in self.ordered_modules], dim=1
            ).norm(dim=1, keepdim=True)
            for name in self.grads:
                self.grads[name] /= norm

    def search(
        self,
        queries: dict[str, Tensor],
        k: int | None,
        modules: set[str] | None = None,
    ):
        """
        Search for the `k` nearest examples in the index based on the query or queries.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.
            module: The name of the module to search for. If `None`,
                all modules will be searched.

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        q = {
            name: queries[name].to(self.device, self.dtype)
            for name in self.ordered_modules
        }

        if self.unit_norm:
            norm = torch.cat(list(q.values()), dim=1).norm(dim=1, keepdim=True)

            for name in q:
                q[name] /= norm + torch.finfo(norm.dtype).eps

        if self.faiss_index:
            if modules:
                raise NotImplementedError(
                    "FAISS index does not implement module-specific search."
                )

            q = (
                torch.cat([q[name] for name in self.ordered_modules], dim=1)
                .cpu()
                .numpy()
            )

            distances, indices = self.faiss_index.search(q, k)

            return torch.from_numpy(distances), torch.from_numpy(indices)

        if modules:
            modules = set([name for name in self.ordered_modules if name in modules])
        else:
            modules = set(self.ordered_modules)

        k = min(k or self.N, self.N)

        scores = torch.stack(
            [q[name] @ self.grads[name].mT for name in modules], dim=-1
        ).sum(-1)

        return torch.topk(scores, k)

    @contextmanager
    def trace(
        self,
        module: nn.Module,
        k: int | None,
        *,
        precondition: bool = False,
        modules: set[str] | None = None,
    ) -> Generator[TraceResult, None, None]:
        """
        Context manager to trace the gradients of a module and return the
        corresponding Attributor instance.
        """
        mod_grads = defaultdict(list)
        result = TraceResult()

        def callback(name: str, g: Tensor):
            g = g.flatten(1)

            # Precondition the gradient using Cholesky solve
            if precondition:
                eigval, eigvec = self.processor.preconditioners_eigen[name]
                eigval_inverse_sqrt = 1.0 / (eigval).sqrt()
                P = eigvec * eigval_inverse_sqrt @ eigvec.mT
                g = g.type_as(P)
                g = g @ P

            # Store the gradient for later use
            mod_grads[name].append(g.to(self.device, self.dtype, non_blocking=True))

        with GradientCollector(module, callback, self.processor, modules):
            yield result

        if not mod_grads:
            raise ValueError("No grads collected. Did you forget to call backward?")

        queries = {
            name: torch.cat(mod_grads[name], dim=1)
            for name in self.ordered_modules
            if name in mod_grads
        }

        if any(q.isnan().any() for q in queries.values()):
            raise ValueError("NaN found in queries.")

        result._scores, result._indices = self.search(queries, k, modules)
