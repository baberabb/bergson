from pathlib import Path

import pytest
import torch
from datasets import Dataset

from bergson import (
    GradientCollector,
    GradientProcessor,
    MemmapScoreWriter,
    collect_gradients,
)
from bergson.data import QueryConfig
from bergson.scorer import build_scorer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_query(tmp_path: Path, model, dataset):
    processor = GradientProcessor(projection_dim=16)
    shapes = GradientCollector(model.base_model, lambda x: x, processor).shapes()

    query_gradient_ds = Dataset.from_list(
        [
            {module: torch.randn(shape.numel()) for module, shape in shapes.items()}
            for _ in range(2)
        ]
    ).with_format("torch", columns=list(shapes.keys()))

    scorer = build_scorer(
        query_gradient_ds,
        QueryConfig(
            query_path=str(tmp_path / "query_gradient_ds"),
            modules=list(shapes.keys()),
            score="mean",
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
        module_wise=False,
        accumulate_grads="mean",
        normalize_accumulated_grad=True,
    )
    score_writer = MemmapScoreWriter(
        scorer,
        len(dataset),
        tmp_path,
        rank=0,
        modules=list(shapes.keys()),
        module_wise=False,
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=processor,
        path=tmp_path,
        score_writer=score_writer,
        module_wise=False,
    )

    assert any(tmp_path.iterdir()), "Expected artifacts in the temp run_path"
    assert any(Path(tmp_path).glob("scores.bin")), "Expected scores file"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_module_wise_query(tmp_path: Path, model, dataset):
    processor = GradientProcessor(projection_dim=16)
    shapes = GradientCollector(model.base_model, lambda x: x, processor).shapes()

    query_gradient_ds = Dataset.from_list(
        [
            {module: torch.randn(shape.numel()) for module, shape in shapes.items()}
            for _ in range(2)
        ]
    ).with_format("torch", columns=list(shapes.keys()))

    scorer = build_scorer(
        query_gradient_ds,
        QueryConfig(
            query_path=str(tmp_path / "query_gradient_ds"),
            modules=list(shapes.keys()),
            score="mean",
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
        module_wise=True,
        accumulate_grads="mean",
        normalize_accumulated_grad=True,
    )
    score_writer = MemmapScoreWriter(
        scorer,
        len(dataset),
        tmp_path,
        rank=0,
        modules=list(shapes.keys()),
        module_wise=True,
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=processor,
        path=tmp_path,
        score_writer=score_writer,
        module_wise=True,
    )

    assert any(tmp_path.iterdir()), "Expected artifacts in the temp run_path"
    assert any(tmp_path.glob("scores.bin")), "Expected scores file"
