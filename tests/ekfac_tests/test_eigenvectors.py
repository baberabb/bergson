import os
from typing import Literal

import torch
from safetensors.torch import load_file

from bergson.approx_unrolling.utils import TensorDict


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
    diff = ground_truth_eigenvectors.sub(run_eigenvectors).abs()
    # Compare if eigenvectors are close, need to use a different check here, since eigenvectors can be off by a sign
    product = ground_truth_eigenvectors.matmul(
        run_eigenvectors.transpose(-1, -2)
    ).abs()  # this should be close to identity matrix

    identity = TensorDict({k: torch.eye(v.shape[-1], device=v.device, dtype=v.dtype) for k, v in product.items()})
    # more lax tolerances for gradient eigenvectors as they have noise
    atol = 1e-5 if eigenvector_type == "activation" else 0.5

    equal_dict = identity.allclose(product, atol=atol)

    if all(equal_dict.values()):
        print(f"{eigenvector_type} eigenvectors match!")

    else:
        diff = product.sub(identity).abs()
        max_diff = diff.max()
        # print keys for which the covariances do not match
        print(f"{eigenvector_type} eigenvectors do not match!")
        for k, v in equal_dict.items():
            if not v:
                print(f"Covariance {k} does not match with absolute difference {max_diff[k]:.3f}")

    print("-*" * 50)
