import pytest

from bergson.data import load_gradients

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

if not HAS_CUDA:
    pytest.skip(
        "Skipping GPU-only tests: no CUDA/NVIDIA driver available.",
        allow_module_level=True,
    )

from pathlib import Path

from transformers import AutoModelForCausalLM

from bergson import (
    AttentionConfig,
    GradientProcessor,
    collect_gradients,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_split_attention_build(tmp_path: Path, model, dataset):
    attention_cfgs = {
        "h.0.attn.attention.out_proj": AttentionConfig(
            num_heads=16, head_size=4, head_dim=2
        ),
    }

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(projection_dim=16),
        path=str(tmp_path),
        attention_cfgs=attention_cfgs,
    )

    assert any(tmp_path.iterdir()), "Expected artifacts in the temp run_path"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_conv1d_build(tmp_path: Path, dataset):
    model_name = "openai-community/gpt2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, use_safetensors=True
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(),
        path=str(tmp_path),
        # This build hangs in pytest with preconditioners enabled.
        # It works when run directly so it may be a pytest issue.
        skip_preconditioners=True,
    )

    assert any(tmp_path.iterdir()), "Expected artifacts in the run path"

    index = load_gradients(str(tmp_path))

    assert len(modules := index.dtype.names) != 0
    assert len(index[modules[0]]) == len(dataset)
    assert index[modules[0]][0].sum().item() != 0.0
