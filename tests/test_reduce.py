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
        ],
        cwd=tmp_path,
        capture_output=True,
        # Get strings instead of bytes
        text=True,
    )

    assert (
        "error" not in result.stderr.lower()
    ), f"Error found in stderr: {result.stderr}"
