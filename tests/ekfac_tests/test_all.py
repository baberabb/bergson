import argparse
import json
import os

import torch
from test_covariance import test_covariances
from test_eigenvalue_correction import test_eigenvalue_correction
from test_eigenvectors import test_eigenvectors

from bergson.data import DataConfig, IndexConfig
from bergson.hessians.compute_EKFAC import compute_EKFAC

parser = argparse.ArgumentParser(description="Run EKFAC tests.")

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

args = parser.parse_args()


test_dir = args.test_dir
overwrite = args.overwrite
use_fsdp = args.use_fsdp
world_size = args.world_size


ground_truth_path = os.path.join(test_dir, "ground_truth")
run_path = os.path.join(test_dir, "run/influence_results")


def test_total_processed_examples():
    total_processed_ground_truth_path = os.path.join(ground_truth_path, "covariances/stats.json")
    total_processed_run_path = os.path.join(run_path, "total_processed.pt")

    with open(total_processed_ground_truth_path, "r") as f:
        ground_truth_data = json.load(f)
        total_processed_ground_truth = ground_truth_data["total_processed_global"]

    total_processed_run = torch.load(total_processed_run_path).item()

    equal = total_processed_ground_truth == total_processed_run

    if equal:
        print(f"Total processed examples match: {total_processed_ground_truth}")
    else:
        print(
            f"Total processed examples do not match! Ground truth: {total_processed_ground_truth},"
            f" Run: {total_processed_run}"
        )
    print("-*" * 50)


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

    if not os.path.exists(run_path) or overwrite:
        compute_EKFAC(cfg=cfg)
        print("EKFAC computation completed successfully.")
    else:
        print("Using existing run directory.")

    test_total_processed_examples()

    test_covariances(
        run_path=run_path,
        ground_truth_path=ground_truth_path,
        covariance_type="activation",
    )
    test_covariances(
        run_path=run_path,
        ground_truth_path=ground_truth_path,
        covariance_type="gradient",
    )

    # Currently this tests for close equality, but does not account for sign differences in eigenvectors. TODO: fix.
    test_eigenvectors(
        run_path=run_path,
        ground_truth_path=ground_truth_path,
        eigenvector_type="activation",
    )
    test_eigenvectors(
        run_path=run_path,
        ground_truth_path=ground_truth_path,
        eigenvector_type="gradient",
    )

    test_eigenvalue_correction(ground_truth_path=ground_truth_path, run_path=run_path)
    print("\n \n All tests done \n \n")


if __name__ == "__main__":
    main()
