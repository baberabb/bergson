import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset
from ml_dtypes import bfloat16
from transformers import AutoConfig, AutoModelForCausalLM

from bergson import (
    GradientProcessor,
    collect_gradients,
)
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig, ScoreConfig
from bergson.data import create_index, load_scores
from bergson.score.score import precondition_ds
from bergson.score.score_writer import MemmapScoreWriter
from bergson.score.scorer import Scorer
from bergson.utils.utils import (
    assert_type,
    convert_precision_to_torch,
    get_gradient_dtype,
    tensor_to_numpy,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_gradients_query(tmp_path: Path, dataset):
    # Create index for uncompressed gradients from a large model.
    config = AutoConfig.from_pretrained(
        "EleutherAI/pythia-1.4b", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_config(config)

    collector = GradientCollector(
        model.base_model, data=dataset, cfg=IndexConfig(run_path=str(tmp_path))
    )
    grad_sizes = {name: math.prod(s) for name, s in collector.shapes().items()}

    dataset.save_to_disk(str(tmp_path / "query_ds" / "data.hf"))
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
            "score",
            "test_score_e2e",
            "--projection_dim",
            "0",
            "--query_path",
            str(tmp_path / "query_ds"),
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
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (
        "error" not in result.stderr.lower()
    ), f"Error found in stderr: {result.stderr}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_score(tmp_path: Path, model, dataset):
    processor = GradientProcessor(projection_dim=16)
    collector = GradientCollector(
        model.base_model,
        data=dataset,
        cfg=IndexConfig(run_path=str(tmp_path)),
        processor=processor,
    )
    shapes = collector.shapes()

    cfg = IndexConfig(run_path=str(tmp_path), token_batch_size=1024)
    score_cfg = ScoreConfig(
        query_path=str(tmp_path / "query_gradient_ds"),
        modules=list(shapes.keys()),
        score="mean",
    )

    query_grads = {
        module: torch.randn(1, shape.numel()) for module, shape in shapes.items()
    }

    score_dtype = (
        convert_precision_to_torch(score_cfg.precision)
        if score_cfg.precision != "auto"
        else get_gradient_dtype(model)
    )

    scorer = Scorer(
        tmp_path,
        len(dataset),
        query_grads,
        score_cfg,
        device=torch.device("cpu"),
        dtype=score_dtype,
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=processor,
        cfg=cfg,
        scorer=scorer,
    )

    assert (tmp_path / "info.json").exists()
    assert (tmp_path / "scores.bin").exists()

    with open(tmp_path / "info.json", "r") as f:
        info = json.load(f)

    scores = load_scores(tmp_path)

    assert len(scores) == len(dataset)

    assert info["num_scores"] == 1

    assert np.allclose(scores["score_0"], np.array([1.8334405, 0.3371131]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_precondition_ds(tmp_path: Path, model, dataset):
    cfg = IndexConfig(run_path=str(tmp_path), token_batch_size=1024)

    preprocess_device = torch.device("cuda:0")

    # Populate and save preconditioners
    processor = GradientProcessor(projection_dim=16)
    collector = GradientCollector(
        model.base_model,
        data=dataset,
        cfg=cfg,
        processor=processor,
    )
    collect_gradients(
        model=model,
        data=dataset,
        processor=processor,
        cfg=cfg,
    )
    processor.save(tmp_path)

    # Produce gradients dataset
    query_ds = Dataset.from_dict(
        {
            module: torch.randn(1, shape.numel())
            for module, shape in collector.shapes().items()
        }
    )

    # Produce preconditioned query dataset
    score_cfg = ScoreConfig(
        query_path=str(tmp_path / "query_gradient_ds"),
        modules=list(collector.shapes().keys()),
        score="mean",
        query_preconditioner_path=str(tmp_path),
    )

    preconditioned_query_ds = precondition_ds(
        query_ds, score_cfg, score_cfg.modules, preprocess_device
    )

    # Produce query dataset without preconditioning
    score_cfg.query_preconditioner_path = None

    vanilla_query_ds = precondition_ds(
        query_ds, score_cfg, score_cfg.modules, preprocess_device
    )

    # Compare the two query datasets
    for name in score_cfg.modules:
        assert not torch.allclose(
            assert_type(torch.Tensor, preconditioned_query_ds[name][:]),
            assert_type(torch.Tensor, vanilla_query_ds[name][:]),
        )


def test_memmap_score_writer_bfloat16(tmp_path: Path):
    """Test that MemmapScoreWriter correctly writes and reads bfloat16 scores."""
    num_items = 10
    num_scores = 3

    writer = MemmapScoreWriter(
        tmp_path, num_items, num_scores, dtype=torch.bfloat16
    )

    # Create some test scores in bfloat16
    scores_batch1 = torch.tensor(
        [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=torch.bfloat16
    )
    scores_batch2 = torch.tensor(
        [[7.5, 8.5, 9.5], [10.5, 11.5, 12.5], [13.5, 14.5, 15.5]],
        dtype=torch.bfloat16,
    )

    # Write scores
    writer([0, 1], scores_batch1)
    writer([5, 6, 7], scores_batch2)
    writer.flush()

    # Verify the files exist
    assert (tmp_path / "scores.bin").exists()
    assert (tmp_path / "info.json").exists()

    # Read back and verify
    with open(tmp_path / "info.json", "r") as f:
        info = json.load(f)

    assert info["num_items"] == num_items
    assert info["num_scores"] == num_scores
    assert "bfloat16" in info["dtype"]["formats"][0]

    # Check written flags
    assert writer.scores["written_0"][0] == True
    assert writer.scores["written_0"][1] == True
    assert writer.scores["written_0"][2] == False  # Not written
    assert writer.scores["written_0"][5] == True
    assert writer.scores["written_0"][6] == True
    assert writer.scores["written_0"][7] == True

    # Check score values (convert back to compare)
    expected_batch1 = tensor_to_numpy(scores_batch1)
    expected_batch2 = tensor_to_numpy(scores_batch2)

    np.testing.assert_array_equal(
        writer.scores["score_0"][[0, 1]].view(bfloat16), expected_batch1[:, 0]
    )
    np.testing.assert_array_equal(
        writer.scores["score_1"][[0, 1]].view(bfloat16), expected_batch1[:, 1]
    )
    np.testing.assert_array_equal(
        writer.scores["score_2"][[0, 1]].view(bfloat16), expected_batch1[:, 2]
    )

    np.testing.assert_array_equal(
        writer.scores["score_0"][[5, 6, 7]].view(bfloat16), expected_batch2[:, 0]
    )


def test_memmap_score_writer_float32(tmp_path: Path):
    """Test that MemmapScoreWriter correctly writes float32 scores (baseline)."""
    num_items = 5
    num_scores = 2

    writer = MemmapScoreWriter(
        tmp_path, num_items, num_scores, dtype=torch.float32
    )

    scores = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)
    writer([0, 1], scores)
    writer.flush()

    # Verify values
    np.testing.assert_array_almost_equal(
        writer.scores["score_0"][[0, 1]], np.array([1.5, 3.5], dtype=np.float32)
    )
    np.testing.assert_array_almost_equal(
        writer.scores["score_1"][[0, 1]], np.array([2.5, 4.5], dtype=np.float32)
    )
