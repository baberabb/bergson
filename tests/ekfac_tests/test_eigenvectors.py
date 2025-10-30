import os
from typing import Literal

import torch
from safetensors.torch import load_file

from bergson.hessians.utils import TensorDict


def test_eigenvectors(
    ground_truth_path,
    run_path,
    eigenvector_type: Literal["activation", "gradient"] = "activation",
):
    eigenvectors_ground_truth_path = os.path.join(
        ground_truth_path, f"eigenvectors/eigenvectors_{eigenvector_type}s.safetensors"
    )
    eigenvectors_run_path = os.path.join(run_path, f"{eigenvector_type}_eigen_sharded")

    # load ground_truth
    ground_truth_eigenvectors = TensorDict(load_file(eigenvectors_ground_truth_path))

    world_size = len(os.listdir(eigenvectors_run_path))  # number of shards
    # load run eigenvectors shards and concatenate them
    run_eigenvectors_shards = [
        os.path.join(eigenvectors_run_path, f"shard_{rank}.safetensors")
        for rank in range(world_size)
    ]
    run_eigenvectors_list = [(load_file(shard)) for shard in run_eigenvectors_shards]
    run_eigenvectors = {}
    for k, v in run_eigenvectors_list[0].items():
        run_eigenvectors[k] = torch.cat(
            [shard[k] for shard in run_eigenvectors_list], dim=0
        )

    run_eigenvectors = TensorDict(run_eigenvectors)

    equal_dict = ground_truth_eigenvectors.allclose(
        run_eigenvectors, atol=0, rtol=1e-10
    )

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
                relative_diff = (
                    100 * max_diff[k] / ground_truth_eigenvectors[k][max_diff_idx].abs()
                )
                if max_diff[k] < 1e-6 and relative_diff < 1e-3:
                    print(f"Eigenvector {k} small differences within tolerance.")
                else:
                    print(
                        f"Eigenvalue corrections {k} does not match with absolute difference {max_diff[k]:.3f} and max "
                        f"rel. difference {relative_diff:.3f} %!"
                    )

                    print("\n")
    print("-*" * 50)
