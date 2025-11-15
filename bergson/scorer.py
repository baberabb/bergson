from pathlib import Path
from typing import Callable

import torch

from .data import QueryConfig
from .score_writer import MemmapScoreWriter, ScoreWriter


class Scorer:
    callback: Callable

    num_scores: int

    writer: ScoreWriter

    device: torch.device

    def __init__(
        self,
        scores_path: Path,
        num_items: int,
        rank: int,
        query_grads: dict[str, torch.Tensor],
        query_cfg: QueryConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype
        self.num_items = num_items
        self.rank = rank

        self.callback = self.build_scorer_callback(
            query_grads,
            query_cfg,
        )

        num_scores = len(query_grads[query_cfg.modules[0]])

        self.writer = MemmapScoreWriter(
            scores_path,
            num_items,
            num_scores,
            rank=rank,
        )

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        scores = self.callback(mod_grads)
        self.writer(indices, scores)

    def build_scorer_callback(
        self,
        query_grads: dict[str, torch.Tensor],
        query_cfg: QueryConfig,
    ) -> Callable:
        """Unified scorer builder for all scorer types."""
        query_tensor = torch.cat(
            [
                query_grads[m].to(device=self.device, dtype=self.dtype)
                for m in query_cfg.modules
            ],
            dim=1,
        )

        @torch.inference_mode()
        def callback(mod_grads: dict[str, torch.Tensor]):
            grads = torch.cat([mod_grads[m] for m in query_cfg.modules], dim=1)
            if query_cfg.unit_normalize:
                grads /= grads.norm(dim=1, keepdim=True)

            if query_cfg.score == "nearest":
                all_scores = grads @ query_tensor.T
                return all_scores.max(dim=-1).values

            return grads @ query_tensor.T

        return callback
