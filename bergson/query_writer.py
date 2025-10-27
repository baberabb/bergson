import csv
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import torch


class Query:
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
