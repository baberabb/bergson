import os
from typing import Literal

import torch
from safetensors.torch import load_file

from bergson.approx_unrolling.utils import TensorDict


def test_covariances(ground_truth_path, run_path, covariance_type: Literal["activation", "gradient"] = "activation"):
    covariances_ground_truth_path = os.path.join(
        ground_truth_path, f"covariances/{covariance_type}_covariance.safetensors"
    )
    covariances_run_path = os.path.join(run_path, f"{covariance_type}_covariance_sharded")

    # load ground_truth
    ground_truth_covariances = TensorDict(load_file(covariances_ground_truth_path))

    world_size = len(os.listdir(covariances_run_path))  # number of shards
    # load run covariances shards and concatenate them
    run_covariances_shards = [
        os.path.join(covariances_run_path, f"shard_{rank}.safetensors") for rank in range(world_size)
    ]
    run_covariances_list = [(load_file(shard)) for shard in run_covariances_shards]
    run_covariances = {}
    for k, v in run_covariances_list[0].items():
        run_covariances[k] = torch.cat([shard[k] for shard in run_covariances_list], dim=0)

    run_covariances = TensorDict(run_covariances)
    diff = ground_truth_covariances.sub(run_covariances).abs()
    # More lax tolerances for gradient covariances as they have noise
    atol = 1e-5 if covariance_type == "activation" else 1e-4
    rtol = 1e-6 if covariance_type == "activation" else 1e-5
    equal_dict = ground_truth_covariances.allclose(run_covariances, atol=atol, rtol=rtol)

    if all(equal_dict.values()):
        print(f"{covariance_type} covariances match")

    else:
        max_diff = diff.max()
        # print keys for which the covariances do not match
        print(f"{covariance_type} covariances do not match!")
        for k, v in equal_dict.items():
            if not v:
                # Find location of max difference
                max_diff_flat_idx = torch.argmax(diff[k])
                max_diff_idx = torch.unravel_index(max_diff_flat_idx, diff[k].shape)
                print(
                    f"Covariance {k} does not match with absolute difference {max_diff[k]:.3f} and "
                    f"relative difference {(100 * max_diff[k] / ground_truth_covariances[k][max_diff_idx].abs()):.3f} %!"
                )
    print("-*" * 50)
