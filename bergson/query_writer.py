import csv
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


class QueryWriter(ABC):
    """
    Base class for query writers.
    """

    @abstractmethod
    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
        **kwargs: Any,
    ):
        """
        Write the scores to the query writer.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def flush(self):
        """
        Flush the query writer.
        """
        raise NotImplementedError("Subclasses must implement this method")


class CsvQueryWriter(QueryWriter):
    """
    Wraps a query scoring callback and stores the resulting scores in a tensor.
    """

    def __init__(
        self,
        query_callback: Callable[..., torch.Tensor],
        num_items: int,
        num_scores: int,
        scores_path: str,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        rank: int,
        module_wise: bool = False,
        rows_per_file: int = 10_000_000,
    ):
        self._query_callback = query_callback
        self._scores_path = Path(scores_path)
        self.rank = rank
        self.dtype = dtype
        self.module_wise = module_wise
        self.num_scores = num_scores
        self.rows_per_file = rows_per_file

        # Find next available CSV file
        self._csv_index = 0
        while (
            os.path.exists(
                self._scores_path / f"rank_{rank}" / f"scores_{self._csv_index:02d}.csv"
            )
            and len(
                pd.read_csv(
                    self._scores_path
                    / f"rank_{rank}"
                    / f"scores_{self._csv_index:02d}.csv"
                )
            )
            >= self.rows_per_file
        ):
            self._csv_index += 1

        self._open_new_csv(self._csv_index)

    def _open_new_csv(self, csv_index: int):
        """Create a new CSV file with a header."""
        self._csv_path = (
            self._scores_path / f"rank_{self.rank}" / f"scores_{csv_index:02d}.csv"
        )
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(self._csv_path):
            print(f"Opening existing CSV file: {self._csv_path}")
            self._rows_in_file = len(pd.read_csv(self._csv_path))
            return

        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["index"] + [f"score_{i}" for i in range(self.num_scores)]
            if self.module_wise:
                header.append("sum_of_squares")
            writer.writerow(header)

        print(f"Started new CSV file: {self._csv_path}")
        self._rows_in_file = 0

    def _maybe_rotate_file(self):
        """Open a new file when reaching the per-file limit."""
        if self._rows_in_file >= self.rows_per_file:
            self._csv_index += 1
            self._open_new_csv(self._csv_index)

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
        name: str | None = None,
    ):
        if name:
            # Accumulate module-wise scores
            scores, sum_of_squares = self._query_callback(mod_grads, name)
            sum_of_squares = sum_of_squares.to(device="cpu", dtype=self.dtype)
            scores = scores.to(device="cpu", dtype=self.dtype)

            if scores.ndim == 1:
                scores = scores.unsqueeze(-1)

            self._write_to_csv_mod(indices, scores, sum_of_squares)
            self._maybe_rotate_file()

        else:
            scores = self._query_callback(mod_grads)
            scores = scores.to(device="cpu", dtype=self.dtype)

            if scores.ndim == 1:
                scores = scores.unsqueeze(-1)

            self._write_to_csv(indices, scores)
            self._maybe_rotate_file()

    def _write_to_csv_mod(
        self, indices: list[int], scores: torch.Tensor, sum_of_squares: torch.Tensor
    ):
        """Write sum_of_squares, scores, and indices to CSV."""
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for idx, score, ss in zip(indices, scores, sum_of_squares):
                row = [idx] + score.tolist() + [ss.item()]
                writer.writerow(row)

    def _write_to_csv(self, indices: list[int], scores: torch.Tensor):
        """Write sum_of_squares, scores, and indices to CSV."""
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for idx, score in zip(indices, scores):
                row = [idx] + score.tolist()
                writer.writerow(row)

    def flush(self):
        pass


class MemmapQueryWriter(QueryWriter):
    """
    Wraps a query scoring callback and stores the resulting scores in a tensor.
    """

    def __init__(
        self,
        query_callback: Callable[..., torch.Tensor],
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
        assert num_scores == 1, "MemmapQueryWriter only supports single-score queries"
        assert module_wise, "MemmapQueryWriter only supports module-wise queries"

        self._query_callback = query_callback
        self._scores_path = Path(scores_path)
        self.rank = rank
        self.dtype = dtype
        self.module_wise = module_wise

        self.flush_interval = flush_batches_interval
        self.num_batches_since_flush = 0

        self.module_to_idx = {mod: i for i, mod in enumerate(modules)}
        self.num_modules = len(modules)

        scores_file_path = os.path.join(scores_path, "scores.bin")

        # Build a json-serializable structured dtype
        struct_dtype = {
            "names": ["index", "score", "sum_of_squares", "module_id", "written"],
            "formats": ["uint32", "float32", "float32", "uint16", "bool"],
            "offsets": [0, 4, 8, 12, 14],
            "itemsize": 16,
        }

        if rank == 0 and not os.path.exists(scores_file_path):
            self.scores = np.memmap(
                str(scores_file_path),
                dtype=np.dtype(struct_dtype),  # type: ignore
                mode="w+",
                shape=(num_items * self.num_modules,),
            )

            # Write zeros
            zeros = np.zeros(len(self.scores), dtype=np.float32)
            self.scores["score"][:] = zeros
            self.scores["sum_of_squares"][:] = zeros
            self.scores["index"][:] = zeros.astype(np.uint32)
            self.scores["module_id"][:] = zeros.astype(np.uint16)
            self.scores["written"][:] = False
            self.flush()

            # Persist metadata for future runs
            with open(scores_path + "/info.json", "w") as f:
                json.dump(
                    {
                        "num_items": num_items,
                        "num_modules": self.num_modules,
                        "dtype": struct_dtype,
                        "module_to_idx": self.module_to_idx,
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
            shape=(num_items * self.num_modules,),
        )

    def _write_to_memmap_mod(
        self,
        indices: list[int],
        scores: torch.Tensor,
        sum_of_squares: torch.Tensor,
        module: str,
    ):
        module_idx = self.module_to_idx[module]
        np_indices = np.array(indices, dtype=np.intp)
        np_indices = np_indices * self.num_modules + module_idx

        self.scores["index"][np_indices] = np.array(indices, dtype=np.uint32)
        self.scores["score"][np_indices] = (
            scores.cpu().numpy().astype(np.float32).flatten()
        )
        self.scores["sum_of_squares"][np_indices] = (
            sum_of_squares.cpu().numpy().astype(np.float32).flatten()
        )
        self.scores["written"][np_indices] = True
        self.scores["module_id"][np_indices] = module_idx

        self.num_batches_since_flush += 1
        if self.num_batches_since_flush >= self.flush_interval:
            self.flush()

    def _write_to_memmap(self, indices: list[int], scores: torch.Tensor):
        self.scores["index"][indices] = np.array(indices, dtype=np.uint32)
        self.scores["score"][indices] = (
            scores.cpu().numpy().astype(np.float32).flatten()
        )
        self.scores["written"][indices] = True

        self.num_batches_since_flush += 1
        if self.num_batches_since_flush >= self.flush_interval:
            self.flush()

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
        name: str | None = None,
    ):
        if name:
            # Accumulate module-wise scores
            scores, sum_of_squares = self._query_callback(mod_grads, name)
            sum_of_squares = sum_of_squares.to(device="cpu", dtype=self.dtype)
            scores = scores.to(device="cpu", dtype=self.dtype)

            if scores.ndim == 1:
                scores = scores.unsqueeze(-1)

            self._write_to_memmap_mod(indices, scores, sum_of_squares, name)

        else:
            scores = self._query_callback(mod_grads)
            scores = scores.to(device="cpu", dtype=self.dtype)

            if scores.ndim == 1:
                scores = scores.unsqueeze(-1)

            self._write_to_memmap(indices, scores)

    def flush(self):
        self.scores.flush()
        self.num_batches_since_flush = 0
