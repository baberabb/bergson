from pathlib import Path

import pytest
import torch

from bergson import (
    GradientCollector,
    GradientProcessor,
    MemmapScoreWriter,
    collect_gradients,
)
from bergson.data import IndexConfig, QueryConfig
from bergson.scorer import get_scorer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_query(tmp_path: Path, model, dataset):
    cfg = IndexConfig(run_path=str(tmp_path))

    cfg.skip_preconditioners = True

    processor = GradientProcessor(projection_dim=16)
    shapes = GradientCollector(model.base_model, lambda x: x, processor).shapes()

    query_grads = {
        module: torch.randn(1, shape.numel()) for module, shape in shapes.items()
    }

    score_writer = MemmapScoreWriter(
        tmp_path,
        len(dataset),
        1,
        rank=0,
    )

    scorer = get_scorer(
        query_grads,
        QueryConfig(
            query_path=str(tmp_path / "query_gradient_ds"),
            modules=list(shapes.keys()),
            score="mean",
        ),
        writer=score_writer,
        device=torch.device("cpu"),
        dtype=torch.float32,
        module_wise=False,
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=processor,
        cfg=cfg,
        scorer=scorer,
    )

    assert any(tmp_path.iterdir()), "Expected artifacts in the temp run_path"
    assert any(Path(tmp_path).glob("scores.bin")), "Expected scores file"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_module_wise_query(tmp_path: Path, model, dataset):
    cfg = IndexConfig(run_path=str(tmp_path))

    cfg.skip_preconditioners = True
    cfg.module_wise = True

    processor = GradientProcessor(projection_dim=16)
    shapes = GradientCollector(model.base_model, lambda x: x, processor).shapes()

    query_grads = {
        module: torch.randn(1, shape.numel()) for module, shape in shapes.items()
    }

    score_writer = MemmapScoreWriter(
        tmp_path,
        len(dataset),
        1,
        rank=0,
    )

    scorer = get_scorer(
        query_grads,
        QueryConfig(
            query_path=str(tmp_path / "query_gradient_ds"),
            modules=list(shapes.keys()),
            score="mean",
        ),
        writer=score_writer,
        module_wise=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=processor,
        cfg=cfg,
        scorer=scorer,
    )

    assert any(tmp_path.iterdir()), "Expected artifacts in the temp run_path"
    assert any(tmp_path.glob("scores.bin")), "Expected scores file"
