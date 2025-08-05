import argparse
import json
import os

import torch
from safetensors.torch import load_file

from bergson.data import DataConfig, IndexConfig, load_gradients
from bergson.distributed import distributed_computing
from bergson.hessians.ekfac_apply import ekfac_apply_worker

parser = argparse.ArgumentParser(description="Run apply EKFAC tests.")

parser.add_argument(
    "--test_dir",
    type=str,
    help="Directory containing test files.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing run directory.",
)
parser.add_argument(
    "--use_fsdp",
    action="store_true",
    help="Use Fully Sharded Data Parallel (FSDP).",
)
parser.add_argument(
    "--world_size",
    type=int,
    default=8,
    help="World size for distributed training.",
)

parser.add_argument(
    "--gradient_path",
    type=str,
    help="Path to the gradient.",
)


args = parser.parse_args()


test_dir = args.test_dir
overwrite = args.overwrite
use_fsdp = args.use_fsdp
world_size = args.world_size


ground_truth_path = os.path.join(test_dir, "ground_truth")
run_path = os.path.join(test_dir, "run/influence_results")


def test_gradients(run_path, ground_truth_path):
    ground_truth = load_file(os.path.join(ground_truth_path, "gradients.safetensors"), device="cuda")
    computed_mmap = load_gradients(run_path)

    for k in ground_truth.keys():
        ground_truth_tensor = ground_truth[k].to(dtype=torch.float32)

        computed_tensor = (
            torch.from_numpy(computed_mmap[k].copy()).to(device="cuda").view(-1, *ground_truth_tensor.shape[1:])
        ).to(dtype=torch.float32)

        if not (ground_truth_tensor.shape == computed_tensor.shape):
            raise ValueError(f"Shape mismatch for key {k}: {ground_truth_tensor.shape} vs {computed_tensor.shape}")

        if not torch.allclose(ground_truth_tensor, computed_tensor, rtol=1e-5, atol=float("inf")):
            max_diff = torch.max(torch.abs(ground_truth_tensor - computed_tensor)).item()
            mean_diff = torch.mean(torch.abs(ground_truth_tensor - computed_tensor)).item()
            print(f"Gradient mismatch for key {k}: max diff {max_diff}, mean diff {mean_diff}")

    print("Gradients test done")


def main():
    # assert covariances, eigenvalue_corrections, eigenvectors and index_config.json exist

    required_files = [
        "covariances",
        "eigenvalue_corrections",
        "eigenvectors",
        "index_config.json",
    ]

    for file_name in required_files:
        assert os.path.exists(os.path.join(ground_truth_path, file_name)), f"Missing required file: {file_name}"

    cfg_json = json.load(open(os.path.join(ground_truth_path, "index_config.json"), "r"))

    cfg = IndexConfig(**cfg_json)

    cfg.data = DataConfig(**(cfg_json["data"]))

    cfg.run_path = test_dir + "/run"
    cfg.debug = True
    cfg.fsdp = use_fsdp
    cfg.world_size = world_size
    cfg.ekfac = True
    cfg.gradient_path = args.gradient_path

    if not os.path.exists(run_path) or overwrite:
        distributed_computing(
            cfg=cfg,
            worker_fn=ekfac_apply_worker,
            setup_data=False,
            setup_model=False,
            setup_processor=False,
        )

        print("EKFAC application completed successfully.")
    else:
        print("Using existing run directory.")

    test_gradients(
        run_path=cfg.gradient_path + "_ekfac",
        ground_truth_path=test_dir + "/test_gradients" + "/gradients_after_ekfac",
    )

    print("\n \n All tests done \n \n")


if __name__ == "__main__":
    main()
