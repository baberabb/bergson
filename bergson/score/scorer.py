from pathlib import Path

import torch

from bergson.config import ScoreConfig
from bergson.score.score_writer import MemmapScoreWriter, ScoreWriter


class Scorer:
    writer: ScoreWriter

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

        self.query_tensor = torch.cat(
            [
                query_grads[m].to(device=self.device, dtype=self.dtype)
                for m in score_cfg.modules
            ],
            dim=1,
        )
        self.score_cfg = score_cfg

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
        # Convert the gradients to the scoring dtype
        if next(iter(mod_grads.values())).dtype != self.dtype:
            mod_grads = {name: grad.to(self.dtype) for name, grad in mod_grads.items()}

        scores = self.score(mod_grads)

        self.writer(indices, scores)

    @torch.inference_mode()
    def score(self, mod_grads: dict[str, torch.Tensor]):
        grads = torch.cat([mod_grads[m] for m in self.score_cfg.modules], dim=1)
        if self.score_cfg.unit_normalize:
            grads /= grads.norm(dim=1, keepdim=True)

        if self.score_cfg.score == "nearest":
            all_scores = grads @ self.query_tensor.T
            return all_scores.max(dim=-1).values

        return grads @ self.query_tensor.T
