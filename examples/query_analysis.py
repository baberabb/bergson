import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import Dataset, IterableDataset, load_from_disk

from bergson.data import DataConfig, IndexConfig, load_data_string


def parse_args():
    parser = ArgumentParser(
        description=(
            "Aggregate static query results assuming six max-accumulated queries "
            "and one mean-based query."
        )
    )
    parser.add_argument(
        "--trial-dir",
        type=Path,
        required=True,
        help="Directory containing the saved static query results (static_query.hf).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HF dataset identifier matching the index build.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split used for the index (default: train).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset used for the index, if any.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Flag indicating the index dataset was streamed.",
    )
    parser.add_argument(
        "--run-path",
        type=str,
        default="static-query",
        help="Run path label to instantiate IndexConfig (not used for IO).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to keep per strategy (defaults to k).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write strategy outputs (defaults to trial directory).",
    )
    parser.add_argument(
        "--raw-name",
        type=str,
        default="static_query.hf",
        help="Name of the saved raw search dataset inside the trial directory.",
    )
    return parser.parse_args()


def build_index_config(args) -> IndexConfig:
    data_cfg = DataConfig(dataset=args.dataset, split=args.split, subset=args.subset)
    return IndexConfig(run_path=args.run_path, data=data_cfg, streaming=args.streaming)


def save_strategy(name: str, dataset: Dataset, output_root: Path):
    strategy_dir = output_root / name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    csv_path = strategy_dir / f"{name}.csv"
    dataset.to_csv(csv_path)

    hf_path = strategy_dir / "hf"
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

    trial_dir = args.trial_dir
    output_root = args.output_dir or trial_dir
    raw_path = trial_dir / args.raw_name

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Static query results not found at {raw_path}. "
            "Run bergson.static_query.query_gradient_dataset first."
        )

    raw_ds = load_from_disk(raw_path)
    scores_tensor = torch.tensor(raw_ds["scores"], dtype=torch.float32)
    indices_tensor = torch.tensor(raw_ds["indices"], dtype=torch.int64)

    if scores_tensor.ndim != 2 or indices_tensor.ndim != 2:
        raise ValueError(
            "Expected scores and indices to have shape (num_queries, k). "
            f"Got {scores_tensor.shape} and {indices_tensor.shape}."
        )

    num_queries, num_neighbors = scores_tensor.shape
    if num_queries != 7:
        raise ValueError(
            f"Expected exactly 7 queries (6 for max, 1 for mean); "
            f"received {num_queries}."
        )

    limit = args.limit or num_neighbors

    max_scores_by_index: dict[int, float] = {}
    for idx, score in zip(
        indices_tensor[1:].reshape(-1).tolist(),
        scores_tensor[1:].reshape(-1).tolist(),
    ):
        idx = int(idx)
        score = float(score)
        best = max_scores_by_index.get(idx)
        if best is None or score > best:
            max_scores_by_index[idx] = score

    sorted_max = sorted(
        max_scores_by_index.items(), key=lambda item: item[1], reverse=True
    )[:limit]
    max_indices = [idx for idx, _ in sorted_max]
    max_scores = [score for _, score in sorted_max]

    mean_indices = indices_tensor[0].tolist()[:limit]
    mean_scores = scores_tensor[0].tolist()[:limit]

    index_cfg = build_index_config(args)
    max_ds = get_text_rows(max_indices, max_scores, index_cfg, score_column="max_score")
    mean_ds = get_text_rows(
        mean_indices, mean_scores, index_cfg, score_column="mean_score"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    max_csv, max_hf = save_strategy("max_strategy", max_ds, output_root)
    mean_csv, mean_hf = save_strategy("mean_strategy", mean_ds, output_root)

    print(f"Saved max strategy outputs to {max_csv} and {max_hf}")
    print(f"Saved mean strategy outputs to {mean_csv} and {mean_hf}")


if __name__ == "__main__":
    main()
