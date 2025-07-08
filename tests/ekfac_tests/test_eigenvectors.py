import os
from typing import Literal

import torch
from safetensors.torch import load_file

from bergson.hessians.utils import TensorDict


def test_eigenvectors(ground_truth_path, run_path, eigenvector_type: Literal["activation", "gradient"] = "activation"):
    eigenvectors_ground_truth_path = os.path.join(
        ground_truth_path, f"eigenvectors/eigenvectors_{eigenvector_type}s.safetensors"
    )
    eigenvectors_run_path = os.path.join(run_path, f"{eigenvector_type}_eigen_sharded")

    # load ground_truth
    ground_truth_eigenvectors = TensorDict(load_file(eigenvectors_ground_truth_path))

    world_size = len(os.listdir(eigenvectors_run_path))  # number of shards
    # load run eigenvectors shards and concatenate them
    run_eigenvectors_shards = [
        os.path.join(eigenvectors_run_path, f"shard_{rank}.safetensors") for rank in range(world_size)
    ]
    run_eigenvectors_list = [(load_file(shard)) for shard in run_eigenvectors_shards]
    run_eigenvectors = {}
    for k, v in run_eigenvectors_list[0].items():
        run_eigenvectors[k] = torch.cat([shard[k] for shard in run_eigenvectors_list], dim=0)

    run_eigenvectors = TensorDict(run_eigenvectors)

    equal_dict = ground_truth_eigenvectors.allclose(run_eigenvectors, rtol=1e-5)

    if all(equal_dict.values()):
        print(f"{eigenvector_type} eigenvectors match!")

    else:
        diff = run_eigenvectors.sub(ground_truth_eigenvectors).abs()
        max_diff = diff.max()
        # print keys for which the covariances do not match
        print(f"{eigenvector_type} eigenvectors do not match!")
        for k, v in equal_dict.items():
            if not v:
                # Find location of max difference
                max_diff_flat_idx = torch.argmax(diff[k])
                max_diff_idx = torch.unravel_index(max_diff_flat_idx, diff[k].shape)
                print(
                    f"Eigenvalue corrections {k} does not match with absolute difference {max_diff[k]:.3f} and "
                    f"relative difference {(100 * max_diff[k] / ground_truth_eigenvectors[k][max_diff_idx].abs()):.3f} %!"
                )
                print(
                    "max difference",
                    ground_truth_eigenvectors[k][max_diff_idx],
                    run_eigenvectors[k][max_diff_idx],
                )
                print(
                    "total abs sum: ",
                    "ground truth",
                    ground_truth_eigenvectors[k].abs().sum(),
                    "run",
                    run_eigenvectors[k].abs().sum(),
                )
                print("\n")
    print("-*" * 50)
