from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM

from bergson import (
    AttentionConfig,
)
from bergson.collection import collect_gradients
from bergson.data import IndexConfig, load_gradients
from bergson.gradients import GradientProcessor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_e2e(tmp_path: Path):
    result = subprocess.run(
        [
            "python",
            "-m",
            "bergson",
            "build",
            "test_e2e",
            "--model",
            "EleutherAI/pythia-14m",
            "--dataset",
            "NeelNanda/pile-10k",
            "--split",
            "train[:100]",
            "--truncation",
        ],
        cwd=tmp_path,
    )

    assert result.returncode == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_consistency(tmp_path: Path, model, dataset):
    cfg = IndexConfig(run_path=str(tmp_path))
    cfg.skip_preconditioners = True
    print(str(tmp_path))
    kwargs = {
        "model": model,
        "data": dataset,
        "processor": GradientProcessor(),
        "cfg": cfg,
    }

    collect_gradients(**kwargs)

    index = load_gradients(cfg.partial_run_path)

    # Regenerate cache
    cache_path = Path("runs/test_build_cache.npy")
    if not cache_path.exists():
        np.save(cache_path, index[index.dtype.names[0]][0])
    cached_item_grad = np.load(cache_path)

    first_module_grad = index[index.dtype.names[0]][0]

    assert np.allclose(first_module_grad, cached_item_grad, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_split_attention_build(tmp_path: Path, model, dataset):
    attention_cfgs = {
        "h.0.attn.attention.out_proj": AttentionConfig(
            num_heads=16, head_size=4, head_dim=2
        ),
    }

    cfg = IndexConfig(run_path=str(tmp_path))

    kwargs = {
        "model": model,
        "data": dataset,
        "processor": GradientProcessor(projection_dim=16),
        "cfg": cfg,
        "attention_cfgs": attention_cfgs,
    }
    collect_gradients(**kwargs)

    assert any(
        Path(cfg.partial_run_path).iterdir()
    ), "Expected artifacts in the temp run_path"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_conv1d_build(tmp_path: Path, dataset):
    model_name = "openai-community/gpt2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, use_safetensors=True
    )

    cfg = IndexConfig(run_path=str(tmp_path))

    cfg.skip_preconditioners = True
    kwargs = {
        "model": model,
        "data": dataset,
        "processor": GradientProcessor(projection_dim=16),
        "cfg": cfg,
    }
    collect_gradients(**kwargs)

    assert any(
        Path(cfg.partial_run_path).iterdir()
    ), "Expected artifacts in the run path"

    index = load_gradients(cfg.partial_run_path)

    assert len(modules := index.dtype.names) != 0
    assert len(index[modules[0]]) == len(dataset)
    assert index[modules[0]][0].sum().item() != 0.0
