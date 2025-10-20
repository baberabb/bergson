import os

import torch
from safetensors.torch import load_file

from bergson.hessians.utils import TensorDict


def test_eigenvalue_correction(ground_truth_path, run_path):
    """Test eigenvalue corrections by comparing to ground truth."""

    lambda_ground_truth_path = os.path.join(
        ground_truth_path, "eigenvalue_corrections/eigenvalue_corrections.safetensors"
    )
    lambda_run_path = os.path.join(run_path, "eigenvalue_correction_sharded")

    # load ground_truth
    lambda_ground_truth = TensorDict(load_file(lambda_ground_truth_path))

    world_size = len(os.listdir(lambda_run_path))  # number of shards
    lambda_run_shards_path = [os.path.join(lambda_run_path, f"shard_{rank}.safetensors") for rank in range(world_size)]
    lambda_list_shards = [(load_file(shard_path)) for shard_path in lambda_run_shards_path]
    lambda_run = {}
    for k, v in lambda_list_shards[0].items():
        if len(v.shape) == 0:
            lambda_run[k] = v
        else:
            lambda_run[k] = torch.cat([shard[k] for shard in lambda_list_shards], dim=0)

    lambda_run = TensorDict(lambda_run)

    total_processed_run_path = os.path.join(run_path, "total_processed_lambda_correction.pt")
    total = torch.load(total_processed_run_path).to(device=lambda_run[list(lambda_run.keys())[0]].device)
    lambda_run.div_(total)
    rtol = 1e-10
    equal_dict = lambda_ground_truth.allclose(lambda_run, rtol=rtol)

    if all(equal_dict.values()):
        print("Eigenvalue corrections match!")
    else:
        print("Eigenvalue corrections do not match!")

        diff = lambda_ground_truth.sub(lambda_run).div(lambda_ground_truth).abs()
        max_diff = diff.max()
        # print keys for which the covariances do not match

        for k, v in equal_dict.items():
            if not v:
                # Find location of max difference

                coord = diff[k].argmax()
                a, b = coord // lambda_ground_truth[k].shape[1], coord % lambda_ground_truth[k].shape[1]

                print(
                    f"Eigenvalue correction {k} does not match with max relative difference "
                    f"{(100 * max_diff[k]):.3f} % and mean {100 * diff[k].mean()} % !",
                )
                print(
                    f"Max relative difference at index {a},{b} with ground truth value "
                    f"{lambda_ground_truth[k][a, b]:.3e} and run value {lambda_run[k][a, b]:.3e}"
                )

                print("\n")
