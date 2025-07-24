import json
import os
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Generator

import faiss
import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from torch import Tensor, nn

from .data import load_unstructured_gradients
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


def gradients_loader(root_dir: str):
    def load_shard(shard_dir: str) -> np.memmap:
        print("shard dir", shard_dir)
        with open(os.path.join(shard_dir, "info.json")) as f:
            info = json.load(f)

        if "grad_size" in info:
            return load_unstructured_gradients(shard_dir)

        dtype = info["dtype"]
        num_grads = info["num_grads"]

        return np.memmap(
            os.path.join(shard_dir, "gradients.bin"),
            dtype=dtype,
            mode="r",
            shape=(num_grads,),
        )

    root_path = Path(root_dir)
    if (root_path / "info.json").exists():
        yield load_shard(root_dir)
    else:
        for shard_path in sorted(root_path.iterdir()):
            if shard_path.is_dir():
                yield load_shard(str(shard_path))


def normalize_grads(grads: np.ndarray) -> np.ndarray:
    batch_size = 1024
    grads_t = torch.tensor(grads)

    for i in range(0, grads_t.shape[0], batch_size):
        batch = grads_t[i : i + batch_size].to(device="cuda", non_blocking=True)
        batch /= batch.norm(dim=1, keepdim=True)
        grads_t[i : i + batch_size] = batch.cpu()

    return grads_t.numpy()


class Attributor:
    def __init__(
        self,
        index_path: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        unit_norm: bool = True,
        batch_size: int = 1024,
        faiss_cfg: str = "Flat",
        max_index_size: int = 700_000,
    ):
        """
        [Guidelines on building your FAISS configuration string](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

        Common configurations:
        - "Flat": exact nearest neighbors with brute force search.
        - "PQ16": nearest neighbors with PQ16 compression. Reduces memory usage.
        - "IVF1024,Flat": approximate nearest neighbors with IVF1024 clustering.
            Enables faster queries at the cost of a slower initial index build.

        GPU indexes will be sharded across GPUs.
        """
        max_chunks = 100

        path = (
            Path("runs/faiss")
            / Path(index_path).stem
            / f"{faiss_cfg.replace(',', '_')}_{max_chunks}"
        )
        path.mkdir(exist_ok=True, parents=True)

        dl = gradients_loader(index_path)

        start = time()
        buffer = []
        i = 0
        for chunk_idx, grads in enumerate(dl):
            # TODO remove
            if chunk_idx >= max_chunks:
                break

            if grads.dtype.names is not None:
                grads = structured_to_unstructured(grads)
            np_dtype = np.array(torch.tensor([], dtype=dtype)).dtype

            if sum(item.shape[0] for item in buffer) + grads.shape[0] < max_index_size:
                buffer.append(grads)
                continue

            # Save buffer to file
            if (path / f"{i}.index").exists():
                print(f"Index shard already exists: {path / f'{i}.index'}")
            else:
                index = faiss.index_factory(
                    buffer[0].shape[1], faiss_cfg, faiss.METRIC_INNER_PRODUCT
                )

                if device != "cpu":
                    gpus = (
                        list(range(torch.cuda.device_count()))
                        if device == "cuda"
                        else [int(str(device).split(":")[1])]
                    )

                    options = faiss.GpuMultipleClonerOptions()
                    options.shard = True
                    index = faiss.index_cpu_to_gpus_list(index, options, gpus=gpus)

                print("Building FAISS index...")
                buffer = [grads.astype(np_dtype) for grads in buffer]
                index.train(buffer[0])
                index.add(np.concatenate(buffer, axis=0))
                buffer = []

                print(f"Trained index in {time() - start:.2f} seconds.")
                print("Saving index shard...")
                faiss.write_index(
                    faiss.index_gpu_to_cpu(index), str(path / f"{i}.index")
                )

            # Start new buffer
            buffer = [grads]
            i += 1

        self.path = path
        self.device = device
        self.dtype = dtype
        self.faiss_cfg = faiss_cfg

        # Load the gradient processor
        self.processor = GradientProcessor.load(index_path, map_location=device)

    def search(
        self, queries: Tensor, k: int, nprobe: int = 10
    ) -> tuple[Tensor, Tensor]:
        """
        Search for the `k` nearest examples in the index based on the query or queries.
        If fewer than `k` examples are found FAISS will return items with the index -1
        and the maximum negative distance.

        Args:
            queries: The query tensor of shape [..., d].
            k: The number of nearest examples to return for each query.
            nprobe: The number of FAISS vector clusters to search

        Returns:
            A namedtuple containing the top `k` indices and inner products for each
            query. Both have shape [..., k].
        """
        start = time()
        q = queries / queries.norm(dim=1, keepdim=True)
        q = q.to("cpu", non_blocking=True).numpy()

        # Load and fetch from each index shard in turn
        shard_distances = []
        shard_indices = []

        options = faiss.GpuMultipleClonerOptions()
        options.shard = True

        gpus = []
        if self.device != "cpu":
            gpus = (
                list(range(torch.cuda.device_count()))
                if self.device == "cuda"
                else [int(str(self.device).split(":")[1])]
            )

        for shard_path in sorted(self.path.iterdir()):
            print(f"Loading index shard {shard_path}...")
            index = faiss.read_index(str(shard_path))

            if self.device != "cpu":
                index = faiss.index_cpu_to_gpus_list(index, options, gpus=gpus)

            distances, indices = index.search(q, k)
            shard_distances.append(distances)
            shard_indices.append(indices)

        # Get the top-k indices and distances for each query
        distances = np.concatenate(shard_distances, axis=1)
        indices = np.concatenate(shard_indices, axis=1)

        # Get the top-k indices and distances for each query
        top_k_indices = np.argsort(distances, axis=1)[:, :k]
        top_k_distances = distances[np.arange(len(q))[:, None], top_k_indices]

        print(f"Searched in {time() - start:.2f} seconds.")
        return torch.from_numpy(top_k_distances), torch.from_numpy(top_k_indices)

    @contextmanager
    def trace(
        self,
        module: nn.Module,
        k: int,
        *,
        precondition: bool = False,
        unit_norm: bool = True,
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
            mod_grads.append(g.to(self.device, self.dtype, non_blocking=True))

        with GradientCollector(module, callback, self.processor):
            yield result

        if not mod_grads:
            raise ValueError("No grads collected. Did you forget to call backward?")

        queries = torch.cat(mod_grads, dim=1)

        if unit_norm:
            queries /= queries.norm(dim=1, keepdim=True)

        result._scores, result._indices = self.search(queries, k)
