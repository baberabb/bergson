import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from bergson import (
    GradientCollector,
    GradientProcessor,
    collect_gradients,
)
from bergson.data import IndexConfig, QueryConfig, create_index
from bergson.scorer import Scorer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_gradients_query(tmp_path: Path, dataset):
    # Create index for uncompressed gradients from a large model.
    config = AutoConfig.from_pretrained(
        "EleutherAI/pythia-1.4b", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_config(config)
    collector = GradientCollector(model.base_model, lambda x: x)
    grad_sizes = {name: math.prod(s) for name, s in collector.shapes().items()}

    dataset.save_to_disk(tmp_path / "query_ds" / "data.hf")
    create_index(
        tmp_path / "query_ds",
        num_grads=len(dataset),
        grad_sizes=grad_sizes,
        dtype=np.float32,
        with_structure=False,
    )

    result = subprocess.run(
        [
            "python",
            "-m",
            "bergson",
            "query",
            "test_query_e2e",
            "--projection_dim",
            "0",
            "--query_path",
            str(tmp_path / "query_ds"),
            "--scores_path",
            str(tmp_path / "scores"),
            "--model",
            "EleutherAI/pythia-1.4b",
            "--dataset",
            "NeelNanda/pile-10k",
            "--split",
            "train[:8]",
            "--truncation",
            "--token_batch_size",
            "256",
            "--skip_preconditioners",
            "--save_index",
            "false",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Error" not in result.stderr, f"Error found in stderr:\n{result.stderr}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_query(tmp_path: Path, model, dataset):
    cfg = IndexConfig(run_path=str(tmp_path))

    processor = GradientProcessor(projection_dim=16)
    shapes = GradientCollector(model.base_model, lambda x: x, processor).shapes()

    query_grads = {
        module: torch.randn(1, shape.numel()) for module, shape in shapes.items()
    }

    dtype = model.dtype if model.dtype != "auto" else torch.float32

    scorer = Scorer(
        tmp_path,
        len(dataset),
        query_grads,
        QueryConfig(
            query_path=str(tmp_path / "query_gradient_ds"),
            modules=list(shapes.keys()),
            score="mean",
        ),
        device=torch.device("cpu"),
        dtype=dtype,
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
