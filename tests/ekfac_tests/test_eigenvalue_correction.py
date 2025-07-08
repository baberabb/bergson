import os

import torch
from safetensors.torch import load_file

from bergson.approx_unrolling.utils import TensorDict


def test_eigenvalue_correction(ground_truth_path, run_path):
    """Test eigenvalue corrections by comparing to ground truth."""

    lambda_ground_truth_path = os.path.join(
        ground_truth_path, "eigenvalue_corrections/eigenvalue_corrections.safetensors"
    )
    lambda_run_path = os.path.join(run_path, "eigenvalue_correction_sharded")

    # load ground_truth
    lambda_ground_truth = TensorDict(load_file(lambda_ground_truth_path))

    world_size = len(os.listdir(lambda_run_path))  # number of shards
    # load run covariances shards and concatenate them
    lambda_run_shards_path = [os.path.join(lambda_run_path, f"shard_{rank}.safetensors") for rank in range(world_size)]
    lambda_list_shards = [(load_file(shard_path)) for shard_path in lambda_run_shards_path]
    lambda_run = {}
    for k, v in lambda_list_shards[0].items():
        lambda_run[k] = torch.cat([shard[k] for shard in lambda_list_shards], dim=0)

    lambda_run = TensorDict(lambda_run)

    rtol = 1e-5
    equal_dict = lambda_ground_truth.allclose(lambda_run, rtol=rtol)

    if all(equal_dict.values()):
        print("Eigenvalue corrections match!")
    else:
        print("Eigenvalue corrections do not match!")

        diff = lambda_ground_truth.sub(lambda_run).abs()
        max_diff = diff.max()
        # print keys for which the covariances do not match

        for k, v in equal_dict.items():
            if not v:
                # Find location of max difference
                max_diff_flat_idx = torch.argmax(diff[k])
                max_diff_idx = torch.unravel_index(max_diff_flat_idx, diff[k].shape)
                print(
                    f"Eigenvalue corrections {k} does not match with absolute difference {max_diff[k]:.3f} and "
                    f"relative difference {(100 * max_diff[k] / lambda_ground_truth[k][max_diff_idx].abs()):.3f} %!"
                )
                print(max_diff_idx, lambda_ground_truth[k][max_diff_idx], lambda_run[k][max_diff_idx])
                print(lambda_ground_truth[k].abs().sum(), lambda_run[k].abs().sum())
