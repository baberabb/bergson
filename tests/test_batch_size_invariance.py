"""
Test that gradient scales are invariant to batch size.

This test verifies the fix for issue #112 where gradients would vary in scale
depending on whether data was processed separately or together.
"""

import subprocess
from pathlib import Path

import pytest
import torch
from datasets import Dataset

from bergson.data import load_gradients


@pytest.mark.slow
@pytest.mark.parametrize("batch_size_a,batch_size_b", [(100, 100), (50, 150)])
def test_gradient_scale_invariance(tmp_path, batch_size_a, batch_size_b):
    """
    Test that gradient scales don't depend on how we batch the data.

    This reproduces the bug from issue #112: when computing gradients for the same
    set of datapoints, the gradient magnitudes should be consistent regardless of
    whether we process them separately or together.

    The fix changes loss.mean().backward() to loss.sum().backward() to make
    gradient scales invariant to batch size.
    """
    # Create two simple datasets
    texts_a = [f"The quick brown fox jumps over the lazy dog {i}" for i in range(batch_size_a)]
    texts_b = [f"A journey of a thousand miles begins with a single step {i}" for i in range(batch_size_b)]

    ds_a = Dataset.from_dict({"text": texts_a})
    ds_b = Dataset.from_dict({"text": texts_b})
    ds_combined = Dataset.from_dict({"text": texts_a + texts_b})

    # Save datasets
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ds_a.save_to_disk(str(data_dir / "data_a"))
    ds_b.save_to_disk(str(data_dir / "data_b"))
    ds_combined.save_to_disk(str(data_dir / "data_combined"))

    # Build three indices with minimal settings for speed
    index_dir = tmp_path / "indices"
    index_dir.mkdir()

    def run_bergson_build(index_name: str, dataset_path: str):
        index_path = index_dir / index_name
        cmd = [
            "bergson", "build", str(index_path),
            "--model", "gpt2",  # Use small model for testing
            "--dataset", dataset_path,
            "--prompt_column", "text",
            "--projection_dim", "8",  # Small for speed
            "--token_batch_size", "1000",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return index_path

    # Build indices
    index_a = run_bergson_build("a", str(data_dir / "data_a"))
    index_b = run_bergson_build("b", str(data_dir / "data_b"))
    index_combined = run_bergson_build("combined", str(data_dir / "data_combined"))

    # Load gradients
    grads_a = torch.from_numpy(
        load_gradients(index_a, structured=False).copy()
    ).float()
    grads_b = torch.from_numpy(
        load_gradients(index_b, structured=False).copy()
    ).float()
    grads_combined = torch.from_numpy(
        load_gradients(index_combined, structured=False).copy()
    ).float()

    # Split combined to match a and b
    grads_a_in_combined = grads_combined[:batch_size_a]
    grads_b_in_combined = grads_combined[batch_size_a:]

    # Compute standard deviations
    std_a_sep = grads_a.std().item()
    std_a_comb = grads_a_in_combined.std().item()
    std_b_sep = grads_b.std().item()
    std_b_comb = grads_b_in_combined.std().item()

    # With the fix (sum instead of mean), the standard deviations should be very close
    # We allow 20% tolerance to account for numerical noise and outliers
    ratio_a = std_a_sep / std_a_comb if std_a_comb > 0 else float('inf')
    ratio_b = std_b_sep / std_b_comb if std_b_comb > 0 else float('inf')

    # Before the fix, these ratios could be 6x or more different
    # After the fix, they should be close to 1.0
    assert 0.8 <= ratio_a <= 1.2, (
        f"Gradient scales for dataset A differ too much between separate and combined: "
        f"ratio = {ratio_a:.2f}x (std_sep={std_a_sep:.2e}, std_comb={std_a_comb:.2e})"
    )
    assert 0.8 <= ratio_b <= 1.2, (
        f"Gradient scales for dataset B differ too much between separate and combined: "
        f"ratio = {ratio_b:.2f}x (std_sep={std_b_sep:.2e}, std_comb={std_b_comb:.2e})"
    )

    # Also check that cosine similarity is high (gradients point in the same direction)
    a_norm = grads_a / grads_a.norm(dim=1, keepdim=True)
    a_comb_norm = grads_a_in_combined / grads_a_in_combined.norm(dim=1, keepdim=True)
    cosines = (a_norm * a_comb_norm).sum(dim=1)

    assert cosines.mean() > 0.99, (
        f"Gradients should point in the same direction: cosine similarity = {cosines.mean():.4f}"
    )
