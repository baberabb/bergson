import math
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig
from bergson.data import create_index, load_gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_gradients_build(tmp_path: Path, dataset):
    # Create index for uncompressed gradients from a large model.
    config = AutoConfig.from_pretrained(
        "EleutherAI/pythia-1.4b", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_config(config)
    collector = GradientCollector(
        model=model.base_model, data=dataset, cfg=IndexConfig(run_path=str(tmp_path))
    )
    grad_sizes = {name: math.prod(s) for name, s in collector.shapes().items()}

    create_index(
        tmp_path,
        num_grads=len(dataset),
        grad_sizes=grad_sizes,
        dtype=np.float32,
        with_structure=False,
    )

    # Load a large gradient index without structure.
    load_gradients(tmp_path, structured=False)

    with pytest.raises(ValueError):
        # Max item size exceeded.
        load_gradients(tmp_path, structured=True)
