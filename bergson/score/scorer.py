from pathlib import Path
from typing import Callable

import torch

from bergson.config import ScoreConfig
from bergson.score.score_writer import MemmapScoreWriter, ScoreWriter


class Scorer:
    scorer_callback: Callable

    num_scores: int

    writer: ScoreWriter

    device: torch.device

    def __init__(
        self,
        path: Path,
        num_items: int,
        query_grads: dict[str, torch.Tensor],
        score_cfg: ScoreConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype
        self.num_items = num_items

        self.scorer_callback = self.build_scorer_callback(
            query_grads,
            score_cfg,
        )

        num_scores = len(query_grads[score_cfg.modules[0]])

        self.writer = MemmapScoreWriter(
            path,
            num_items,
            num_scores,
        )

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        first_grad = next(iter(mod_grads.values()))
        if first_grad.dtype != self.dtype:
            mod_grads = {name: grad.to(self.dtype) for name, grad in mod_grads.items()}

        scores = self.scorer_callback(mod_grads)
        self.writer(indices, scores)

    def build_scorer_callback(
        self,
        query_grads: dict[str, torch.Tensor],
        score_cfg: ScoreConfig,
    ) -> Callable:
        """Unified scorer builder for all scorer types."""
        query_tensor = torch.cat(
            [
                query_grads[m].to(device=self.device, dtype=self.dtype)
                for m in score_cfg.modules
            ],
            dim=1,
        )

        @torch.inference_mode()
        def callback(mod_grads: dict[str, torch.Tensor]):
            grads = torch.cat([mod_grads[m] for m in score_cfg.modules], dim=1)
            if score_cfg.unit_normalize:
                grads /= grads.norm(dim=1, keepdim=True)

            if score_cfg.score == "nearest":
                all_scores = grads @ query_tensor.T
                return all_scores.max(dim=-1).values

            return grads @ query_tensor.T

        return callback
