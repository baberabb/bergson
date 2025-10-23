import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import Dataset, IterableDataset, concatenate_datasets, load_from_disk

from bergson.data import DataConfig, IndexConfig, load_data_string


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--scores_path",
        type=Path,
        required=True,
        help="Directory containing the query scores (scores.hf).",
    )
    parser.add_argument(
        "--dataset", type=str, default="EleutherAI/deep-ignorance-pretraining-mix"
    )
    parser.add_argument("--split", type=str, default="train")

    return parser.parse_args()


def save_strategy(name: str, dataset: Dataset, output_root: Path):
    strategy_dir = output_root / name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    csv_path = strategy_dir / f"{name}.csv"
    dataset.to_csv(csv_path)

    hf_path = strategy_dir / "data.hf"
    if hf_path.exists():
        shutil.rmtree(hf_path)
    dataset.save_to_disk(hf_path, num_proc=1)

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
    do_query_normalization = True

    args = parse_args()

    limit = 50

    file_name = "scores.hf"
    col_name = "scores"

    sum_of_squares_file_name = "scores_sum_of_squares.hf"
    sum_of_squares_col_name = "sum_of_squares"

    # file_name = "scores_normalized_scores.hf"
    # col_name = "normalized_scores"

    query_path = "runs/wmdp_bio_robust_mcqa_means_0/full_accum_mean_mod_grads.hf"
    query_scores_path = Path(args.scores_path)
    assert query_scores_path.exists(), f"Query scores not found at {query_scores_path}."

    if (query_scores_path / "shard-00000").exists():
        datasets = []
        ss_datasets = []
        for shard_path in sorted(query_scores_path.iterdir()):
            if shard_path.is_dir() and "shard" in shard_path.name:
                datasets.append(load_from_disk(shard_path / file_name))
                ss_datasets.append(
                    load_from_disk(shard_path / sum_of_squares_file_name)
                )
        scores_ds = concatenate_datasets(datasets)
        ss_ds = concatenate_datasets(ss_datasets)
    else:
        scores_ds = load_from_disk(query_scores_path / file_name)
        ss_ds = load_from_disk(query_scores_path / sum_of_squares_file_name)

    scores_tensor = torch.tensor(scores_ds[col_name], dtype=torch.float32)
    ss_tensor = torch.tensor(ss_ds[sum_of_squares_col_name], dtype=torch.float32)
    values_norm = ss_tensor.sqrt()

    # breakpoint()
    try:
        indices_tensor = torch.tensor(scores_ds["indices"], dtype=torch.int64)
    except Exception as e:
        print(f"Error loading indices: {e}")
        indices_tensor = torch.tensor(range(len(scores_ds)), dtype=torch.int64)
        indices_tensor = indices_tensor.unsqueeze(-1)
        print(indices_tensor.shape)

    # ss_tensor = ss_tensor.unsqueeze(-1)
    # batch_size = 1024
    # for i in range(0, len(ss_tensor), batch_size):
    # batch = ss_tensor[i:i+batch_size]
    # scores_tensor[i:i+batch_size] =
    # scores_tensor[i:i+batch_size] / (batch.sqrt() + 1e-12)
    if do_query_normalization:
        query_ds = load_from_disk(query_path)
        query_ds = query_ds.with_format("torch", columns=list(query_ds.column_names))
        query_norm = (
            torch.stack(
                [(query_ds[module][:] ** 2).sum() for module in query_ds.column_names]
            )
            .sum()
            .sqrt()
        )
        scores_tensor /= (query_norm * (values_norm + 1e-12)).unsqueeze(-1)
    else:
        scores_tensor = scores_tensor / (values_norm + 1e-12)

    print(f"max score: {scores_tensor.max()}, min score: {scores_tensor.min()}")

    assert scores_tensor.ndim == 2 and indices_tensor.ndim == 2, (
        f"Expected scores and indices to have shape (num_queries, k). "
        f"Got {scores_tensor.shape} and {indices_tensor.shape}."
    )

    # num_scores_per_doc, k = scores_tensor.shape
    # if num_scores_per_doc != 7:
    #     raise ValueError(
    #         f"Expected exactly 7 queries (6 for max, 1 for mean); "
    #         f"received {num_scores_per_doc} for each of {k} top documents."
    #     )
    # if num_scores_per_doc != 1:
    #     raise ValueError(
    #         f"Expected exactly 1 query (for mean); "
    #         f"received {num_scores_per_doc} for each of {k} top documents."
    #     )

    # Skip first row, flatten remaining 6 rows
    # scores_flat = scores_tensor[1:].reshape(-1)  # Shape: (6 * num_neighbors,)
    # indices_flat = indices_tensor[1:].reshape(-1)  # Shape: (6 * num_neighbors,)

    # # Use scatter_reduce to get max score for each unique index
    # max_doc_idx = int(indices_flat.max().item() + 1)
    # max_scores = torch.full((max_doc_idx,), float("-inf"), dtype=torch.float32)
    # max_scores.scatter_reduce_(0, indices_flat, scores_flat, reduce="amax")

    # # Convert to dict, excluding indices that were never seen
    # max_scores_by_index = {
    #     idx: score.item()
    #     for idx, score in enumerate(max_scores)
    #     if score != float("-inf")
    # }

    # sorted_max = sorted(
    #     max_scores_by_index.items(), key=lambda item: item[1], reverse=True
    # )
    # max_indices = [idx for idx, _ in sorted_max]
    # max_scores = [score for _, score in sorted_max]

    mean_indices = indices_tensor[:, 0]  # .tolist()
    mean_scores = scores_tensor[:, 0]  # .tolist()

    top_k = min(
        limit, scores_tensor.shape[0]
    )  # Handle case where there are fewer than 50
    top_scores, top_positions = torch.topk(mean_scores, k=top_k, largest=True)

    top_k_indices = mean_indices[top_positions].tolist()
    top_k_scores = top_scores.tolist()

    print("top k scores", top_k_scores)

    mean_indices = top_k_indices
    mean_scores = top_k_scores

    print(
        f"max score: {torch.tensor(mean_scores).max()}, "
        f"min score: {torch.tensor(mean_scores).min()}"
    )

    # TODO get the top limit
    # if limit is not None:
    # max_indices = max_indices[:limit]
    #     max_scores = max_scores[:limit]
    # mean_indices = mean_indices[:limit]
    # mean_scores = mean_scores[:limit]

    data_cfg = DataConfig(dataset=args.dataset, split=args.split)
    index_cfg = IndexConfig(run_path="", data=data_cfg, streaming=True)

    # max_ds =
    # get_text_rows(max_indices, max_scores, index_cfg, score_column="max_score")
    mean_ds = get_text_rows(
        mean_indices, mean_scores, index_cfg, score_column="mean_score"
    )

    query_scores_path.mkdir(parents=True, exist_ok=True)
    # max_csv, max_hf = save_strategy("max", max_ds, query_scores_path)
    mean_csv, mean_hf = save_strategy("mean", mean_ds, query_scores_path)

    # print(f"Saved max top results to {max_csv} and {max_hf}")
    print(f"Saved mean top results to {mean_csv} and {mean_hf}")


if __name__ == "__main__":
    main()
