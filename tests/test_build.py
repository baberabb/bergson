from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM

from bergson import (
    AttentionConfig,
    GradientProcessor,
    collect_gradients,
)
from bergson.data import load_gradients


def test_build_consistency(tmp_path: Path, model, dataset):
    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(),
        path=str(tmp_path),
        skip_preconditioners=True,
    )
    index = load_gradients(str(tmp_path))

    # Regenerate cache
    cache_path = Path("runs/test_build_cache.npy")
    if not cache_path.exists():
        np.save(cache_path, index[index.dtype.names[0]][0])
    cached_item_grad = np.load(cache_path)

    first_module_grad = index[index.dtype.names[0]][0]

    assert np.allclose(first_module_grad, cached_item_grad, atol=1e-6)


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
