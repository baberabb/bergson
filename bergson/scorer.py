from typing import Callable

import torch

from .data import QueryConfig
from .score_writer import ScoreWriter


class Scorer:
    callback: Callable

    num_scores: int

    module_wise: bool

    module_wise_scores: dict[str, torch.Tensor]

    module_wise_ssq: dict[str, torch.Tensor]

    writer: ScoreWriter

    def __init__(
        self,
        callback: Callable,
        num_scores: int,
        writer: ScoreWriter,
        module_wise: bool = False,
    ):
        self.callback = callback
        self.num_scores = num_scores
        self.writer = writer
        self.module_wise = module_wise
        self.module_wise_scores = {}
        self.module_wise_ssq = {}

    def __call__(
        self, indices: list[int], mod_grads: dict[str, torch.Tensor], **kwargs
    ):
        if self.module_wise:
            name = kwargs["name"]
            scores, ssq = self.callback(mod_grads, name=name)

            self.module_wise_scores[name] = scores
            if ssq is not None:
                self.module_wise_ssq[name] = ssq
        else:
            scores = self.callback(mod_grads, **kwargs)
            self.writer(indices, scores)

    def finalize_module_wise(self, indices: list[int]):
        if not self.module_wise:
            return

        # Accumulate an inner product for each index
        # [num_items, num_scores, num_modules] -> [num_items, num_scores]
        scores = torch.stack(
            [scores for scores in self.module_wise_scores.values()], dim=-1
        ).sum(dim=-1)

        # Normalize with the sum of squares if present
        if self.module_wise_ssq:
            # [num_items, num_scores, num_modules] -> [num_items, num_scores]
            sum_of_squares = torch.stack(
                [sum_of_squares for sum_of_squares in self.module_wise_ssq.values()],
                dim=-1,
            ).sum(dim=-1)
            assert scores.shape[0] == sum_of_squares.shape[0]
            scores *= sum_of_squares.rsqrt()

        self.writer(indices, scores)


def build_scorer_callback(
    query_grads: dict[str, torch.Tensor],
    query_cfg: QueryConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    module_wise: bool = False,
    nearest: bool = False,
) -> Callable:
    """Unified scorer builder for all scorer types."""
    if not module_wise:
        query_tensor = torch.cat(
            [query_grads[m].to(device=device, dtype=dtype) for m in query_cfg.modules],
            dim=1,
        )
    else:
        query_grads = {name: grad.to(dtype) for name, grad in query_grads.items()}
        query_tensor = None

    @torch.inference_mode()
    def callback(mod_grads: dict[str, torch.Tensor], **kwargs):
        if query_tensor is None:
            name = kwargs["name"]
            module_scores = mod_grads[name] @ query_grads[name].T

            # Save sum of squares for post-hoc normalization
            if query_cfg.unit_normalize:
                mod_grads[name].pow_(2)
                ssq = mod_grads[name].sum(dim=1)
            else:
                ssq = None

            return module_scores, ssq

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
    module_wise: bool,
    device: torch.device,
    dtype: torch.dtype,
):
    num_scores = len(query_grads[query_cfg.modules[0]])

    if query_cfg.score == "mean":
        return Scorer(
            build_scorer_callback(
                query_grads,
                query_cfg,
                device,
                dtype,
                module_wise=module_wise,
            ),
            num_scores,
            writer,
            module_wise=module_wise,
        )
    elif query_cfg.score == "nearest":
        assert not module_wise, "Module-wise scoring not supported for nearest query"
        return Scorer(
            build_scorer_callback(
                query_grads,
                query_cfg,
                device,
                dtype,
                module_wise=False,
                nearest=True,
            ),
            num_scores,
            writer,
            module_wise=module_wise,
        )
    elif query_cfg.score == "individual":
        return Scorer(
            build_scorer_callback(
                query_grads,
                query_cfg,
                device,
                dtype,
                module_wise=module_wise,
            ),
            num_scores,
            writer,
            module_wise=module_wise,
        )
    else:
        raise ValueError(f"Invalid scoring method: {query_cfg.score}")
