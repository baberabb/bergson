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
        modules: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.unit_norm = unit_norm
        self.faiss_index = None

        # Load the gradient processor
        self.processor = GradientProcessor.load(index_path, map_location=device)

        # Load the gradient index
        if faiss_cfg:
            self.faiss_index = FaissIndex(index_path, faiss_cfg, device, unit_norm)
            self.N = self.faiss_index.ntotal
        else:
            mmap = load_gradients(index_path)

            # Copy gradients into device memory
            self.grads = {
                name: torch.tensor(mmap[name], device=device, dtype=dtype)
                for name in mmap.dtype.names
            }
            self.N = mmap[mmap.dtype.names[0]].shape[0]

            if unit_norm:
                norm = torch.cat([grad for grad in self.grads.values()], dim=1).norm(
                    dim=1, keepdim=True
                )
                for name in self.grads:
                    self.grads[name] /= norm

    def search(
        self, queries: dict[str, Tensor], k: int, modules: list[str] | None = None
    ) -> tuple[Tensor, Tensor]:
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
        k = min(k, self.N)

        scores = torch.stack(
            [q[name] @ self.grads[name].mT for name in modules], dim=-1
        ).sum(-1)

        return torch.topk(scores, k)

    def search_module(
        self, queries: Tensor, k: int, module: str
    ) -> tuple[Tensor, Tensor]:
        """
        Search for the `k` nearest examples in the index based on the query or queries.
        If fewer than `k` examples are found FAISS will return items with the index -1
        and the maximum negative distance.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.
            nprobe: The number of FAISS vector clusters to search if using ANN.

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        assert isinstance(
            self.grads, dict
        ), "Gradients must be a dictionary of tensors."
        assert module in self.grads, f"Module {module} not found in gradients."

        k = min(k, self.grads[module].shape[0])

        q = queries

        if self.unit_norm:
            q /= q.norm(dim=1, keepdim=True)

        if not self.faiss_cfg:
            return torch.topk(q.to(self.device) @ self.grads[module].mT, k)

        q = q.cpu().numpy()

        shard_distances = []
        shard_indices = []
        offset = 0

        for index in self.faiss_shards:
            index.nprobe = self.faiss_cfg.nprobe
            distances, indices = index.search(q, k)

            indices += offset
            offset += index.ntotal

            shard_distances.append(distances)
            shard_indices.append(indices)

        distances = np.concatenate(shard_distances, axis=1)
        indices = np.concatenate(shard_indices, axis=1)

        # Rerank results overfetched from multiple shards
        if len(self.faiss_shards) > 1:
            topk_indices = np.argsort(distances, axis=1)[:, :k]
            indices = indices[np.arange(indices.shape[0])[:, None], topk_indices]
            distances = distances[np.arange(distances.shape[0])[:, None], topk_indices]

        return torch.from_numpy(distances.squeeze()), torch.from_numpy(
            indices.squeeze()
        )

    @contextmanager
    def trace(
        self, module: nn.Module, k: int, *, precondition: bool = False
    ) -> Generator[TraceResult, None, None]:
        """
        Context manager to trace the gradients of a module and return the
        corresponding Attributor instance.
        """
        mod_grads = defaultdict(list)
        result = TraceResult()

        def callback(name: str, g: Tensor):
            # Precondition the gradient using Cholesky solve
            if precondition:
                eigval, eigvec = self.processor.preconditioners_eigen[name]
                eigval_inverse_sqrt = 1.0 / (eigval).sqrt()
                P = eigvec * eigval_inverse_sqrt @ eigvec.mT
                g = g.flatten(1).type_as(P)
                g = g @ P
            else:
                g = g.flatten(1)

            # Store the gradient for later use
            mod_grads[name].append(g.to(self.device, self.dtype, non_blocking=True))

        with GradientCollector(module, callback, self.processor):
            yield result

        if not mod_grads:
            raise ValueError("No grads collected. Did you forget to call backward?")

        queries = {name: torch.cat(g, dim=1) for name, g in mod_grads.items()}

        if any(q.isnan().any() for q in queries.values()):
            raise ValueError("NaN found in queries.")

        result._scores, result._indices = self.search(queries, k)
