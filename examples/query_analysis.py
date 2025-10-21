import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import Dataset, IterableDataset, concatenate_datasets, load_from_disk

from bergson.data import DataConfig, IndexConfig, load_data_string


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--query_scores",
        type=Path,
        required=True,
        help="Directory containing the saved query results (scores.hf).",
    )
    parser.add_argument(
        "--raw_name",
        type=str,
        default="static_query.hf",
        help="Name of the saved raw search dataset inside the trial directory.",
    )
    parser.add_argument(
        "--dataset", type=str, default="EleutherAI/deep-ignorance-pretraining-mix"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true")

    return parser.parse_args()


def save_strategy(name: str, dataset: Dataset, output_root: Path):
    strategy_dir = output_root / name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    csv_path = strategy_dir / f"{name}.csv"
    dataset.to_csv(csv_path)

    hf_path = strategy_dir / "data.hf"
    if hf_path.exists():
        shutil.rmtree(hf_path)
    dataset.save_to_disk(hf_path)

    return csv_path, hf_path


def get_text_rows(
    indices: list[int],
    scores: list[float],
    index_cfg: IndexConfig,
    score_column: str = "score",
):
    ds = load_data_string(
        index_cfg.data.dataset, index_cfg.data.split, streaming=index_cfg.streaming
    )

    if not index_cfg.streaming:
        assert isinstance(ds, Dataset), "Dataset required for direct selection"

        selected = ds.select(indices)
        return selected.add_column(score_column, scores, new_fingerprint="score")

    assert isinstance(ds, IterableDataset), "IterableDataset required for streaming"

    pending_idxs = {idx: score for idx, score in zip(indices, scores)}
    data = []

    for i, row in enumerate(ds):
        if i in pending_idxs:
            score = pending_idxs.pop(i)
            data.append(
                {
                    score_column: score,
                    **row,
                }
            )
            if not pending_idxs:
                break

    assert not pending_idxs, (
        f"Could not collect all rows for the requested indices."
        f" Missing indices: {sorted(pending_idxs)}"
    )

    return Dataset.from_list(data)


def main():
    args = parse_args()

    trial_dir = Path(args.query_scores)

    assert trial_dir.exists(), f"Query scores not found at {trial_dir}."

    if (trial_dir / "shard-00000").exists():
        datasets = []
        for shard_path in sorted(trial_dir.iterdir()):
            if shard_path.is_dir() and "shard" in shard_path.name:
                try:
                    datasets.append(load_from_disk(shard_path / "data.hf"))
                except FileNotFoundError:
                    datasets.append(load_from_disk(shard_path / "scores.hf"))
        scores_ds = concatenate_datasets(datasets)
    else:
        try:
            scores_ds = load_from_disk(trial_dir / "data.hf")
        except FileNotFoundError:
            scores_ds = load_from_disk(trial_dir / "scores.hf")

    scores_tensor = torch.tensor(scores_ds["scores"], dtype=torch.float32)
    indices_tensor = torch.tensor(scores_ds["indices"], dtype=torch.int64)

    if scores_tensor.ndim != 2 or indices_tensor.ndim != 2:
        raise ValueError(
            "Expected scores and indices to have shape (num_queries, k). "
            f"Got {scores_tensor.shape} and {indices_tensor.shape}."
        )

    num_queries, num_docs = scores_tensor.shape
    if num_queries != 7:
        raise ValueError(
            f"Expected exactly 7 queries (6 for max, 1 for mean); "
            f"received {num_queries} for each of {num_docs} documents."
        )

        # Skip first row, flatten remaining 6 rows
    scores_flat = scores_tensor[1:].reshape(-1)  # Shape: (6 * num_neighbors,)
    indices_flat = indices_tensor[1:].reshape(-1)  # Shape: (6 * num_neighbors,)

    # Use scatter_reduce to get max score for each unique index
    num_docs = int(indices_flat.max().item() + 1)
    max_scores = torch.full((num_docs,), float("-inf"), dtype=torch.float32)
    max_scores.scatter_reduce_(0, indices_flat, scores_flat, reduce="amax")

    # Convert to dict, excluding indices that were never seen
    max_scores_by_index = {
        idx: score.item()
        for idx, score in enumerate(max_scores)
        if score != float("-inf")
    }

    sorted_max = sorted(
        max_scores_by_index.items(), key=lambda item: item[1], reverse=True
    )
    max_indices = [idx for idx, _ in sorted_max]
    max_scores = [score for _, score in sorted_max]

    mean_indices = indices_tensor[0].tolist()
    mean_scores = scores_tensor[0].tolist()

    data_cfg = DataConfig(dataset=args.dataset, split=args.split)
    index_cfg = IndexConfig(run_path="", data=data_cfg, streaming=args.streaming)

    max_ds = get_text_rows(max_indices, max_scores, index_cfg, score_column="max_score")
    mean_ds = get_text_rows(
        mean_indices, mean_scores, index_cfg, score_column="mean_score"
    )

    trial_dir.mkdir(parents=True, exist_ok=True)
    max_csv, max_hf = save_strategy("max", max_ds, trial_dir)
    mean_csv, mean_hf = save_strategy("mean", mean_ds, trial_dir)

    print(f"Saved max top results to {max_csv} and {max_hf}")
    print(f"Saved mean top results to {mean_csv} and {mean_hf}")


if __name__ == "__main__":
    main()
