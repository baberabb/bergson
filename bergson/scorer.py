from typing import Callable

import torch

from .data import QueryConfig
from .score_writer import ScoreWriter


class Scorer:
    callback: Callable

    num_scores: int

    writer: ScoreWriter

    device: torch.device

    def __init__(
        self,
        callback: Callable,
        num_scores: int,
        writer: ScoreWriter,
        device: torch.device,
    ):
        self.callback = callback
        self.num_scores = num_scores
        self.writer = writer
        self.device = device

    def __call__(
        self,
        indices: list[int],
        mod_grads: dict[str, torch.Tensor],
    ):
        mod_grads = {name: grad.to(self.device) for name, grad in mod_grads.items()}

        scores = self.callback(mod_grads)
        self.writer(indices, scores)


def build_scorer_callback(
    query_grads: dict[str, torch.Tensor],
    query_cfg: QueryConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    nearest: bool = False,
) -> Callable:
    """Unified scorer builder for all scorer types."""
    query_tensor = torch.cat(
        [query_grads[m].to(device=device, dtype=dtype) for m in query_cfg.modules],
        dim=1,
    )

    @torch.inference_mode()
    def callback(mod_grads: dict[str, torch.Tensor]):
        grads = torch.cat([mod_grads[m] for m in query_cfg.modules], dim=1)
        if query_cfg.unit_normalize:
            grads /= grads.norm(dim=1, keepdim=True)

        if nearest:
            all_scores = grads @ query_tensor.T
            return all_scores.max(dim=-1).values

        return grads @ query_tensor.T

    return callback


def get_scorer(
    query_grads: dict[str, torch.Tensor],
    query_cfg: QueryConfig,
    writer: ScoreWriter,
    device: torch.device,
    dtype: torch.dtype,
):
    """If in-place is True the scorer may modify the input gradients."""

    num_scores = len(query_grads[query_cfg.modules[0]])

    if query_cfg.score == "mean":
        return Scorer(
            build_scorer_callback(
                query_grads,
                query_cfg,
                device,
                dtype,
            ),
            num_scores,
            writer,
            device,
        )
    elif query_cfg.score == "nearest":
        return Scorer(
            build_scorer_callback(
                query_grads,
                query_cfg,
                device,
                dtype,
                nearest=True,
            ),
            num_scores,
            writer,
            device,
        )
    elif query_cfg.score == "individual":
        return Scorer(
            build_scorer_callback(
                query_grads,
                query_cfg,
                device,
                dtype,
            ),
            num_scores,
            writer,
            device,
        )
    else:
        raise ValueError(f"Invalid scoring method: {query_cfg.score}")
