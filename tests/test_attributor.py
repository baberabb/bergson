from pathlib import Path
from typing import Any

import pytest
import torch

from bergson import Attributor, FaissConfig, GradientProcessor, collect_gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attributor(tmp_path: Path, model, dataset):
    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(projection_dim=16),
        path=str(tmp_path),
    )

    attr = Attributor(str(tmp_path), device="cpu", unit_norm=True)

    x = torch.tensor(dataset[0]["input_ids"]).unsqueeze(0)

    with attr.trace(model.base_model, 5) as result:
        model(x, labels=x).loss.backward()
        model.zero_grad()

    assert result.scores[0, 0].item() > 0.99  # Same item
    assert result.scores[0, 1].item() < 0.50  # Different item


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_faiss(tmp_path: Path, model, dataset):
    dtype: Any = model.dtype
    model.to("cuda")
    dtype = torch.float32 if model.dtype == torch.float32 else torch.float16

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(projection_dim=16),
        path=str(tmp_path),
    )

    attr = Attributor(
        str(tmp_path),
        device="cuda",
        unit_norm=True,
        faiss_cfg=FaissConfig(),
        dtype=dtype,
    )

    x = torch.tensor(dataset[0]["input_ids"]).unsqueeze(0).cuda()

    with attr.trace(model.base_model, 2) as result:
        model(x, labels=x).loss.backward()

        model.zero_grad()

    assert result.scores[0, 0].item() > 0.99  # Same item
    assert result.scores[0, 1].item() < 0.50  # Different item
