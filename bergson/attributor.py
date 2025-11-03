from collections import defaultdict
from contextlib import contextmanager
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
        index_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        unit_norm: bool = False,
        faiss_cfg: FaissConfig | None = None,
        processor: GradientProcessor | None = None,
        unstructured: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.unit_norm = unit_norm
        self.faiss_index = None

        # Load the gradient processor
        self.processor = processor or GradientProcessor.load(
            index_path, map_location=device
        )

        # Load the gradient index
        if faiss_cfg:
            self.faiss_index = FaissIndex(index_path, faiss_cfg, device, unit_norm)
            self.N = self.faiss_index.ntotal
        else:
            mmap = load_gradients(index_path)
            self.N = mmap[mmap.dtype.names[0]].shape[0]

            # Copy gradients into device memory
            if unstructured:
                from numpy.lib.recfunctions import structured_to_unstructured
                import numpy as np
                mmap = structured_to_unstructured(mmap).astype(np.float16)
                print("Number of elements:", mmap.shape[0] * mmap.shape[1])
                print(mmap.dtype)
                print(f"RAM required assuming float32: {mmap.shape[0] * mmap.shape[1] * 4 / 1024**3} GB")
                self.grads = torch.from_numpy(mmap)

                if unit_norm:
                    norm = self.grads.norm(dim=1, keepdim=True) + torch.finfo(dtype).eps
                    self.grads /= norm
            else:
                self.grads = {
                    name: torch.tensor(mmap[name], device=device, dtype=dtype)
                    for name in mmap.dtype.names
                }

                if unit_norm:
                    norm = torch.cat([grad for grad in self.grads.values()], dim=1).norm(
                        dim=1, keepdim=True
                    )
                    for name in self.grads:
                        self.grads[name] /= norm

    def search(
        self,
        queries: dict[str, Tensor],
        k: int | None,
        modules: list[str] | None = None,
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
        q = {name: item.to(self.device, self.dtype) for name, item in queries.items()}

        if self.unit_norm:
            norm = torch.cat(list(q.values()), dim=1).norm(dim=1, keepdim=True)

            for name in q:
                q[name] /= norm + 1e-8

        if self.faiss_index:
            if modules:
                raise NotImplementedError(
                    "FAISS index does not implement module-specific search."
                )

            q = torch.cat([q[name] for name in q], dim=1).cpu().numpy()

            distances, indices = self.faiss_index.search(q, k)

            return torch.from_numpy(distances.squeeze()), torch.from_numpy(
                indices.squeeze()
            )

        modules = modules or list(q.keys())
        k = min(k or self.N, self.N)

        scores = torch.stack(
            [q[name] @ self.grads[name].mT for name in modules], dim=-1
        ).sum(-1)

        return torch.topk(scores, k)

    def score(
        self,
        queries: dict[str, Tensor],
        # modules: list[str] | None = None,
        batch_size: int = 1024,
        onload_device: str = "cuda",
    ):
        """
        Search for the `k` nearest examples in the index based on the query or queries.
        Onload shards into VRAM and search.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.
            module: The name of the module to search for. If `None`,
                all modules will be searched.

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        assert not self.faiss_index, "FAISS index does not implement onloaded search."

        q = {name: item.to(self.device, self.dtype) for name, item in queries.items()}

        if self.unit_norm:
            norm = torch.cat(list(q.values()), dim=1).norm(dim=1, keepdim=True)
            for name in q:
                q[name] /= norm + 1e-8

        # modules = modules or list(q.keys())
        k = self.N

        modules = list(q.keys())

        scores = torch.zeros(k, len(q), device=self.device, dtype=self.dtype)

        q_tensor = torch.cat([q[name] for name in modules], dim=1).to(onload_device)
        for i in range(0, self.N, batch_size):
            batch = self.grads[i : i + batch_size].to(onload_device, self.dtype)
            batch_scores = batch @ q_tensor.mT
            scores[i : i + batch_size] = batch_scores.to(self.device)

        return scores

    @contextmanager
    def trace_score(self,
        module: nn.Module,
        k: int | None,
        *,
        precondition: bool = False,
        target_modules: set[str] | None = None,
    ):

        mod_grads = defaultdict(list)
        result = {}

        def callback(name: str, g: Tensor, indices: list[int]):
            # Precondition the gradient using Cholesky solve
            if precondition:
                eigval, eigvec = self.processor.preconditioners_eigen[name]
                # assert not eigval.isnan().any().item() and not eigvec.isnan().any().item()

                eigval_clamped = torch.clamp(eigval.to(torch.float64), min=0.0)
                # assert not eigval_clamped.isnan().any().item(), "eigval_clamped is nan"
                eigval_inverse_sqrt = 1.0 / (
                    (eigval_clamped).sqrt() + torch.finfo(torch.float64).eps
                )

                P = (
                    eigvec.to(eigval_inverse_sqrt.dtype)
                    * eigval_inverse_sqrt
                    @ eigvec.mT.to(eigval_inverse_sqrt.dtype)
                )
                g = g.flatten(1).type_as(P)
                assert not P.isnan().any().item(), "P is nan"
                assert not g.isnan().any().item(), "g is nan"
                g = g @ P
            else:
                g = g.flatten(1)

            # Store the gradient for later use
            mod_grads[name].append(g.to(self.device, self.dtype, non_blocking=True))

        with GradientCollector(module, callback, self.processor, target_modules):
            yield result

        if not mod_grads:
            raise ValueError("No grads collected. Did you forget to call backward?")

        queries = {name: torch.cat(g, dim=1) for name, g in mod_grads.items()}

        if any(q.isnan().any() for q in queries.values()):
            raise ValueError("NaN found in queries.")

        result['scores'] = self.score(queries)


    @contextmanager
    def trace(
        self,
        module: nn.Module,
        k: int | None,
        *,
        precondition: bool = False,
        target_modules: set[str] | None = None,
        score: bool = False,
    ) -> Generator[TraceResult, None, None]:
        """
        Context manager to trace the gradients of a module and return the
        corresponding Attributor instance.
        """
        mod_grads = defaultdict(list)
        result = TraceResult()

        def callback(name: str, g: Tensor, indices: list[int]):
            # Precondition the gradient using Cholesky solve
            if precondition:
                eigval, eigvec = self.processor.preconditioners_eigen[name]
                # assert not eigval.isnan().any().item() and not eigvec.isnan().any().item()

                eigval_clamped = torch.clamp(eigval.to(torch.float64), min=0.0)
                # assert not eigval_clamped.isnan().any().item(), "eigval_clamped is nan"
                eigval_inverse_sqrt = 1.0 / (
                    (eigval_clamped).sqrt() + torch.finfo(torch.float64).eps
                )  #
                # assert not eigval_inverse_sqrt.isnan().any().item()

                # assert not eigval_inverse_sqrt.isnan().any().item(), "eigval_inverse_sqrt is nan after dtype conversion"
                # eigval_inverse_sqrt = eigval_inverse_sqrt.to(eigval.dtype)
                # P = eigvec * eigval_inverse_sqrt @ eigvec.mT
                P = (
                    eigvec.to(eigval_inverse_sqrt.dtype)
                    * eigval_inverse_sqrt
                    @ eigvec.mT.to(eigval_inverse_sqrt.dtype)
                )
                g = g.flatten(1).type_as(P)
                assert not P.isnan().any().item(), "P is nan"
                assert not g.isnan().any().item(), "g is nan"
                g = g @ P
            else:
                g = g.flatten(1)

            # Store the gradient for later use
            mod_grads[name].append(g.to(self.device, self.dtype, non_blocking=True))

        with GradientCollector(module, callback, self.processor, target_modules):
            yield result

        if not mod_grads:
            raise ValueError("No grads collected. Did you forget to call backward?")

        queries = {name: torch.cat(g, dim=1) for name, g in mod_grads.items()}

        if any(q.isnan().any() for q in queries.values()):
            raise ValueError("NaN found in queries.")

        result._scores, result._indices = self.search(queries, k)
