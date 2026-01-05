import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


class ScoreWriter(ABC):
    """
    Base class for score writers.
    """

    @abstractmethod
    def __call__(
        self,
        indices: list[int],
        scores: torch.Tensor,
    ):
        """
        Write the scores to the score writer.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def flush(self):
        """
        Flush the score writer.
        """
        raise NotImplementedError("Subclasses must implement this method")


class MemmapScoreWriter(ScoreWriter):
    """
    Wraps a score scoring callback and stores the resulting scores in a tensor.
    """

    def __init__(
        self,
        path: Path,
        num_items: int,
        num_scores: int,
        *,
        dtype: torch.dtype = torch.float32,
        flush_interval: int = 64,
    ):
        self.path = path
        self.num_scores = num_scores
        self.dtype = dtype
        self.flush_interval = flush_interval

        self.num_batches_since_flush = 0

        self.path.mkdir(parents=True, exist_ok=True)
        scores_file_path = self.path / "scores.bin"

        # Build a json-serializable structured dtype
        names = []
        formats = []
        offsets = []
        for i in range(self.num_scores):
            names.append(f"score_{i}")
            formats.append("float32")
            offsets.append(i * 6)

            names.append(f"written_{i}")
            formats.append("bool")
            offsets.append(i * 6 + 4)

        total_bytes = sum(np.dtype(fmt).itemsize for fmt in formats)
        # Round up to the nearest 8 bytes
        itemsize = ((total_bytes + 7) // 8) * 8

        struct_dtype = {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": itemsize,
        }

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and not scores_file_path.exists():
            print(f"Creating new scores file: {scores_file_path}")

            self.scores = np.memmap(
                str(scores_file_path),
                dtype=np.dtype(struct_dtype),  # type: ignore
                mode="w+",
                shape=(num_items,),
            )

            for name in names:
                if "written" in name:
                    self.scores[name][:] = False
            self.flush()

            # Persist metadata for future runs
            with (path / "info.json").open("w") as f:
                json.dump(
                    {
                        "num_items": num_items,
                        "num_scores": num_scores,
                        "dtype": struct_dtype,
                    },
                    f,
                    indent=2,
                )

        if dist.is_initialized():
            dist.barrier()

        self.scores = np.memmap(
            str(scores_file_path),
            dtype=np.dtype(struct_dtype),  # type: ignore
            mode="r+",
            shape=(num_items,),
        )

    def __call__(self, indices: list[int], scores: torch.Tensor):
        # scores: [num_indices, num_scores]
        for i in range(self.num_scores):
            self.scores[f"score_{i}"][indices] = (
                scores[:, i].to(dtype=torch.float32).cpu().numpy().flatten()
            )
            self.scores[f"written_{i}"][indices] = True

        self.num_batches_since_flush += 1
        if self.num_batches_since_flush >= self.flush_interval:
            self.flush()

    def flush(self):
        self.scores.flush()
        self.num_batches_since_flush = 0
