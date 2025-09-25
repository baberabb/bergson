import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Protocol

import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.typing import NDArray
from tqdm import tqdm


@dataclass
class FaissConfig:
    """Configuration for FAISS index."""

    index_factory: str = "Flat"
    """
    The [FAISS index factory string](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

    Common FAISS factory strings:
        - "IVF1,SQfp16": exact nearest neighbors with brute force search and fp16.
            Valid for CPU or memmapped indices.
        - "IVF1024,SQfp16": approximate nearest neighbors with 1024 cluster centers
            and fp16. Fast approximate queries are produced at the cost of a slower
            initial index build.
        - "PQ6720": nearest neighbors with vector product quantization to 6720 elements.
            Reduces memory usage at the cost of accuracy.
    """
    mmap_index: bool = False
    """Whether to query the gradients on-disk."""
    max_train_examples: int | None = None
    """The maximum number of examples to train the index on.
        If `None`, all examples will be used."""
    batch_size: int = 1024
    """The batch size for pre-processing gradients."""
    num_shards: int = 1
    """The number of shards to build for an index.
        Using more shards reduces peak RAM usage."""
    nprobe: int = 10
    """The number of FAISS vector clusters to search if using ANN."""


class Index(Protocol):
    """Protocol for searchable FAISS index."""

    def search(self, x: NDArray, k: int) -> tuple[NDArray, NDArray]: ...
    @property
    def ntotal(self) -> int: ...
    @property
    def nprobe(self) -> int: ...
    @nprobe.setter
    def nprobe(self, value: int) -> None: ...
    def train(self, x: NDArray) -> None: ...
    def add(self, x: NDArray) -> None: ...


def normalize_grads(
    grads: NDArray,
    device: str,
    batch_size: int,
) -> NDArray:
    normalized_grads = np.zeros_like(grads).astype(grads.dtype)

    for i in range(0, grads.shape[0], batch_size):
        batch = torch.from_numpy(grads[i : i + batch_size]).to(device)
        normalized_grads[i : i + batch_size] = (
            (batch / batch.norm(dim=1, keepdim=True)).cpu().numpy()
        )

    return normalized_grads


def gradients_loader(root_dir: str):
    def load_shard(shard_dir: str) -> np.memmap:
        with open(os.path.join(shard_dir, "info.json")) as f:
            info = json.load(f)

        return np.memmap(
            os.path.join(shard_dir, "gradients.bin"),
            dtype=info["dtype"],
            mode="r",
            shape=(info["num_grads"],),
        )

    root_path = Path(root_dir)
    if (root_path / "info.json").exists():
        yield load_shard(root_dir)
    else:
        for shard_path in sorted(root_path.iterdir()):
            if shard_path.is_dir() and "shard" in shard_path.name:
                yield load_shard(str(shard_path))


def index_to_device(index: Index, device: str) -> Index:
    try:
        import faiss
    except ImportError:
        raise ImportError("Faiss not found, run `pip install faiss-gpu-cu12`...")
    import faiss

    if device != "cpu":
        gpus = (
            list(range(torch.cuda.device_count()))
            if device == "cuda"
            else [int(device.split(":")[1])]
        )

        try:
            options = faiss.GpuMultipleClonerOptions()
        except AttributeError as e:
            raise ImportError(
                "Faiss not found, you may have faiss-cpu installed instead "
                "of faiss-gpu with `pip install faiss-gpu-cu12`..."
            ) from e

        options.shard = True
        return faiss.index_cpu_to_gpus_list(index, options, gpus=gpus)

    return faiss.index_gpu_to_cpu(index)


class FaissIndex:
    """FAISS index."""

    shards: list[Index]

    def __init__(self, path: str, faiss_cfg: FaissConfig, device: str, unit_norm: bool):
        try:
            import faiss
        except ImportError:
            raise ImportError("Faiss not found, run `pip install faiss-gpu-cu12`")
        import faiss

        self.faiss_cfg = faiss_cfg

        faiss_path = (
            Path("runs/faiss")
            / Path(path).stem
            / (
                f"{faiss_cfg.index_factory.replace(',', '_')}"
                f"{'_unit_norm' if unit_norm else ''}"
            )
        )

        if not (faiss_path.exists() and any(faiss_path.iterdir())):
            print("Building FAISS index...")
            start = time()

            faiss_path.mkdir(exist_ok=True, parents=True)

            num_dataset_shards = len(list(Path(path).iterdir()))
            shards_per_index = math.ceil(num_dataset_shards / faiss_cfg.num_shards)

            dl = gradients_loader(path)
            buffer = []
            index_idx = 0

            for grads in tqdm(dl, desc="Loading gradients"):
                grads = structured_to_unstructured(grads)

                if unit_norm:
                    grads = normalize_grads(grads, device, faiss_cfg.batch_size)

                buffer.append(grads)

                if len(buffer) == shards_per_index:
                    # Build index shard
                    print(f"Building shard {index_idx}...")

                    grads = np.concatenate(buffer, axis=0)
                    buffer = []

                    index = faiss.index_factory(
                        grads.shape[1],
                        faiss_cfg.index_factory,
                        faiss.METRIC_INNER_PRODUCT,
                    )
                    index = index_to_device(index, device)
                    train_examples = faiss_cfg.max_train_examples or grads.shape[0]
                    index.train(grads[:train_examples])
                    index.add(grads)

                    # Write index to disk
                    del grads
                    index = index_to_device(index, "cpu")
                    faiss.write_index(index, str(faiss_path / f"{index_idx}.faiss"))

                    index_idx += 1

            if buffer:
                grads = np.concatenate(buffer, axis=0)
                buffer = []
                index = faiss.index_factory(
                    grads.shape[1], faiss_cfg.index_factory, faiss.METRIC_INNER_PRODUCT
                )
                index = index_to_device(index, device)
                index.train(grads)
                index.add(grads)

                # Write index to disk
                del grads
                index = index_to_device(index, "cpu")
                faiss.write_index(index, str(faiss_path / f"{index_idx}.faiss"))

            print(f"Built index in {(time() - start) / 60:.2f} minutes.")
            del buffer, index

        shards = []
        for i in range(faiss_cfg.num_shards):
            shard = faiss.read_index(
                str(faiss_path / f"{i}.faiss"),
                faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
            )
            if not faiss_cfg.mmap_index:
                shard = index_to_device(shard, device)

            shards.append(shard)

        self.shards = shards

    def search(self, q: NDArray, k: int) -> tuple[NDArray, NDArray]:
        """Note: if fewer than `k` examples are found FAISS will return items
        with the index -1 and the maximum negative distance."""
        shard_distances = []
        shard_indices = []
        offset = 0

        for index in self.shards:
            index.nprobe = self.faiss_cfg.nprobe
            distances, indices = index.search(q, k)

            indices += offset
            offset += index.ntotal

            shard_distances.append(distances)
            shard_indices.append(indices)

        distances = np.concatenate(shard_distances, axis=1)
        indices = np.concatenate(shard_indices, axis=1)

        # Rerank results overfetched from multiple shards
        if len(self.shards) > 1:
            topk_indices = np.argsort(distances, axis=1)[:, :k]
            indices = indices[np.arange(indices.shape[0])[:, None], topk_indices]
            distances = distances[np.arange(distances.shape[0])[:, None], topk_indices]

        return distances, indices

    @property
    def ntotal(self) -> int:
        return sum(shard.ntotal for shard in self.shards)

    @property
    def nprobe(self) -> int:
        return self.shards[0].nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        for shard in self.shards:
            shard.nprobe = value
