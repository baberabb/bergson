import json
import os

import torch
from test_covariance import test_covariances
from test_eigenvalue_correction import test_eigenvalue_correction

from bergson.data import DataConfig, IndexConfig
from bergson.hessians.compute_EKFAC import compute_EKFAC

test_dir = "/root/bergson-approx-unrolling/tests/ekfac_tests/test_files/pile_100_examples"
ground_truth_path = os.path.join(test_dir, "ground_truth")
run_path = os.path.join(test_dir, "run/influence_results")
overwrite = True


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
            f"Total processed examples do not match! Ground truth: {total_processed_ground_truth}, Run: {total_processed_run}"
        )
    print("-*" * 50)


def main():
    # assert covariances, eigenvalue_corrections, eigenvectors and index_config.json exist
    use_fsdp = True
    world_size = 8
    required_files = ["covariances", "eigenvalue_corrections", "eigenvectors", "index_config.json"]

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

    test_covariances(run_path=run_path, ground_truth_path=ground_truth_path, covariance_type="activation")
    test_covariances(run_path=run_path, ground_truth_path=ground_truth_path, covariance_type="gradient")

    # Currently this tests for close equality, but does not account for sign differences in eigenvectors. TODO: fix.
    # test_eigenvectors(run_path=run_path, ground_truth_path=ground_truth_path, eigenvector_type="activation")
    # test_eigenvectors(run_path=run_path, ground_truth_path=ground_truth_path, eigenvector_type="gradient")

    test_eigenvalue_correction(ground_truth_path=ground_truth_path, run_path=run_path)


if __name__ == "__main__":
    main()
