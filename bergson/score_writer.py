import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

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
        mod_grads: dict[str, torch.Tensor],
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

    @abstractmethod
    def finalize_module_wise(self, indices: list[int]):
        """
        Finalize the module-wise scores and write to the memmap.
        """
        raise NotImplementedError("Subclasses must implement this method")


class MemmapScoreWriter(ScoreWriter):
    """
    Wraps a score scoring callback and stores the resulting scores in a tensor.
    """

    def __init__(
        self,
        scorer: Callable,
        num_items: int,
        num_scores: int,
        scores_path: str,
        *,
        dtype: torch.dtype = torch.float32,
        rank: int,
        modules: list[str],
        module_wise: bool = False,
        flush_batches_interval: int = 40,
    ):
        self._scorer = scorer
        self._scores_path = Path(scores_path)
        self.rank = rank
        self.dtype = dtype
        self.module_wise = module_wise

        self.flush_interval = flush_batches_interval
        self.num_batches_since_flush = 0

        self.num_modules = len(modules)

        self.num_scores = num_scores

        self._scores_path.mkdir(parents=True, exist_ok=True)
        scores_file_path = self._scores_path / "scores.bin"

        # Build a json-serializable structured dtype
        names = []
        formats = []
        offsets = []
        for i in range(num_scores):
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

        if rank == 0 and not os.path.exists(scores_file_path):
            print(f"Creating new scores file: {scores_file_path}")

            self.scores = np.memmap(
                str(scores_file_path),
                dtype=np.dtype(struct_dtype),  # type: ignore
                mode="w+",
                shape=(num_items * self.num_modules,),
            )

            # Write zeros
            zeros = np.zeros(len(self.scores), dtype=np.float32)
            for name in names:
                if "score" in name:
                    self.scores[name][:] = zeros
                if "written" in name:
                    self.scores[name][:] = False
            self.flush()

            # Persist metadata for future runs
            with open(scores_path + "/info.json", "w") as f:
                json.dump(
                    {
                        "num_items": num_items,
                        "num_modules": self.num_modules,
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
        print(f"Loaded {len(self.scores)} scores from {scores_file_path}")

        self.module_wise_scores = {}
        self.module_wise_sum_squares = {}

    def _write_to_memmap(self, indices: list[int], scores: torch.Tensor):
        print("len indices", len(indices))
        # scores is [len(indices), num_scores]
        for i in range(self.num_scores):
            self.scores[f"score_{i}"][indices] = (
                scores[:, i].cpu().numpy().astype(np.float32).flatten()
            )
            self.scores[f"written_{i}"][indices] = True

        self.num_batches_since_flush += 1
        if self.num_batches_since_flush >= self.flush_interval:
            self.flush()

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
        name: str | None = None,
    ):
        # Module-wise scores
        if name:
            scores, sum_of_squares = self._scorer(mod_grads, name)
            print("scores from cb shape", scores.shape)
            self.module_wise_scores[name] = scores.to(device="cpu", dtype=self.dtype)
            self.module_wise_sum_squares[name] = sum_of_squares.to(
                device="cpu", dtype=self.dtype
            )
        else:
            scores = self._scorer(mod_grads)
            scores = scores.to(device="cpu", dtype=self.dtype)

            self._write_to_memmap(indices, scores)

    def finalize_module_wise(self, indices: list[int]):
        """Finalize the score by accumulating module-wise scores and writing
        to the memmap. Normalize with the sum of squares if needed."""

        # Accumulate scores
        scores = torch.cat(
            [scores for scores in self.module_wise_scores.values()], dim=1
        )

        # Normalize with the sum of squares
        if self.module_wise_sum_squares:
            # [num_modules, num_items, num_scores] -> [num_items, num_scores]
            sum_of_squares = torch.stack(
                [
                    sum_of_squares
                    for sum_of_squares in self.module_wise_sum_squares.values()
                ],
            ).sum(dim=0)
            assert scores.shape[0] == sum_of_squares.shape[0]
            scores *= sum_of_squares.rsqrt()

        # Write accumulated scores
        self._write_to_memmap(indices, scores)

    def flush(self):
        self.scores.flush()
        self.num_batches_since_flush = 0
