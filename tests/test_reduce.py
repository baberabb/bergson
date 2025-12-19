import subprocess
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reduce_e2e(tmp_path: Path):
    result = subprocess.run(
        [
            "python",
            "-m",
            "bergson",
            "reduce",
            "test_reduce_e2e",
            "--model",
            "EleutherAI/pythia-14m",
            "--dataset",
            "NeelNanda/pile-10k",
            "--split",
            "train[:100]",
            "--truncation",
            "--method",
            "mean",
            "--unit_normalize",
            "--skip_preconditioners",
        ],
        cwd=tmp_path,
        capture_output=True,
        # Get strings instead of bytes
        text=True,
    )

    assert (
        "error" not in result.stderr.lower()
    ), f"Error found in stderr: {result.stderr}"

    # Load the gradient index
    from bergson.config import IndexConfig
    from bergson.data import load_gradient_dataset

    index_cfg = IndexConfig(run_path=str(tmp_path / "test_reduce_e2e"))
    ds = load_gradient_dataset(Path(index_cfg.run_path), structured=False)
    assert len(ds) == 1

    grads = torch.tensor(ds["gradients"][:])
    assert not torch.isnan(grads).any()
