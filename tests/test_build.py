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

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import (
    DataConfig,
    GradientProcessor,
    HeadConfig,
    IndexConfig,
    scan_gradients,
)
from bergson.data import tokenize


def test_disk_build_linear(tmp_path: Path):
    run_path = tmp_path / "example_with_heads"
    run_path.mkdir(parents=True, exist_ok=True)

    config = IndexConfig(
        run_path=str(run_path),
        model="RonenEldan/TinyStories-1M",
        data=DataConfig(dataset="NeelNanda/pile-10k", truncation=True),
        head_cfgs={
            "h.0.attn.attention.out_proj": HeadConfig(
                num_heads=16, head_size=4, head_dim=2
            ),
        },
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model, trust_remote_code=True, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    data = load_dataset(config.data.dataset, split="train[:1%]")
    data = data.select(range(8))  # type: ignore

    processor = GradientProcessor(projection_dim=config.projection_dim)

    data = data.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=config.data, tokenizer=tokenizer),
        remove_columns=data.column_names,
    )

    scan_gradients(
        model=model,
        data=data,
        processor=processor,
        path=config.run_path,
        head_cfgs=config.head_cfgs,
        drop_columns=config.drop_columns,
    )

    assert any(run_path.iterdir()), "Expected artifacts in the temp run_path"


def test_disk_build_conv1d(tmp_path: Path):
    run_path = tmp_path / "example_with_heads"
    run_path.mkdir(parents=True, exist_ok=True)

    config = IndexConfig(
        run_path=str(run_path),
        model="openai-community/gpt2",
        data=DataConfig(dataset="NeelNanda/pile-10k", truncation=True),
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model, trust_remote_code=True, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    data = load_dataset(config.data.dataset, split="train")
    data = data.select(range(8))  # type: ignore

    processor = GradientProcessor(projection_dim=config.projection_dim)

    data = data.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=config.data, tokenizer=tokenizer),
        remove_columns=data.column_names,
    )

    collect_gradients(
        model=model,
        data=data,
        processor=processor,
        path=config.run_path,
        head_cfgs=config.head_cfgs,
    )

    assert any(run_path.iterdir()), "Expected artifacts in the temp run_path"

    index = load_gradients(str(run_path))
    assert len(modules := index.dtype.names) != 0
    assert len(first_column := index[modules[0]]) != 0