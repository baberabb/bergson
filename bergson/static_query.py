import json
import os
from time import perf_counter
from pathlib import Path

import torch
from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer

from bergson import Attributor, FaissConfig
from .data import (
    IndexConfig,
    QueryConfig,
    load_data_string,
    load_gradient_dataset,
    tokenize,
    load_gradients,
)
from .gradients import GradientProcessor


def get_query_data(query_cfg: QueryConfig):
    """
    Load and optionally precondition the query dataset. Preconditioners
    may be mixed as described in https://arxiv.org/html/2410.17413v1#S3.
    """
    # Collect the query gradients if they don't exist
    if not os.path.exists(query_cfg.query_path):
        raise FileNotFoundError(
            f"Query dataset not found at {query_cfg.query_path}. "
            "Please build a query dataset index first."
        )

    # Load the query dataset
    with open(os.path.join(query_cfg.query_path, "info.json"), "r") as f:
        target_modules = json.load(f)["dtype"]["names"]

    query_ds = load_gradient_dataset(query_cfg.query_path, concatenate_gradients=False)
    query_ds = query_ds.with_format(
        "torch", columns=target_modules, format_kwargs={"dtype": torch.float64}
    )

    use_q = query_cfg.query_preconditioner_path is not None
    use_i = query_cfg.index_preconditioner_path is not None

    if use_q or use_i:
        q, i = {}, {}
        if use_q:
            assert query_cfg.query_preconditioner_path is not None
            q = GradientProcessor.load(
                query_cfg.query_preconditioner_path,
                map_location="cuda",
            ).preconditioners
        if use_i:
            assert query_cfg.index_preconditioner_path is not None
            i = GradientProcessor.load(
                query_cfg.index_preconditioner_path, map_location="cuda"
            ).preconditioners

        mixed_preconditioner = (
            {
                k: q[k] * query_cfg.mixing_coefficient
                + i[k] * (1 - query_cfg.mixing_coefficient)
                for k in q
            }
            if (q and i)
            else (q or i)
        )
        mixed_preconditioner = {
            k: v.cuda().to(torch.float64) for k, v in mixed_preconditioner.items()
        }

        def precondition(batch):
            for name in target_modules:
                breakpoint()
                print(batch[name].shape, mixed_preconditioner[name].shape)
                batch[name] = (
                    batch[name].cuda().to(torch.float64) @ mixed_preconditioner[name]
                ).cpu()

            return batch

        breakpoint()
        query_ds = query_ds.map(
            precondition, batched=True, batch_size=query_cfg.batch_size
        )

    return query_ds


def get_text_rows(indices, scores, index_cfg: IndexConfig):
    ds = load_data_string(
        index_cfg.data.dataset, index_cfg.data.split, streaming=index_cfg.streaming
    )


    if not index_cfg.streaming:
        assert isinstance(ds, Dataset), "Dataset required for direct selection"
        return ds.select(indices)
    else:
        rows = []
        assert isinstance(ds, IterableDataset), "IterableDataset required for streaming"
        # Loop through the dataset and collect the indices
        for i, row in enumerate(ds):
            if i in indices:
                rows.append(row)
        return Dataset.from_list(rows)


@torch.inference_mode()
def query_gradient_dataset(
    query_cfg: QueryConfig, index_cfg: IndexConfig, device="cpu", k: int | None = 50
):
    # In many cases the token_batch_size may be smaller than the max length allowed by
    # the model. If cfg.data.truncation is True, we use the tokenizer to truncate
    tokenizer = AutoTokenizer.from_pretrained(
        index_cfg.model, revision=index_cfg.revision
    )
    tokenizer.model_max_length = min(
        tokenizer.model_max_length, index_cfg.token_batch_size
    )

    # # Do all the data loading and preprocessing on the main process
    # ds = load_data_string(
    #     index_cfg.data.dataset, index_cfg.data.split, streaming=index_cfg.streaming
    # )

    # remove_columns = ds.column_names if index_cfg.drop_columns else None
    # ds = ds.map(
    #     tokenize,
    #     batched=True,
    #     fn_kwargs=dict(args=index_cfg.data, tokenizer=tokenizer),
    #     remove_columns=remove_columns,
    # )
    # if index_cfg.data.reward_column:
    #     assert isinstance(ds, Dataset), "Dataset required for advantage estimation"
    #     ds = ds.add_column(
    #         "advantage",
    #         estimate_advantage(ds, index_cfg.data),
    #         new_fingerprint="advantage",  # type: ignore
    #     )

    if not query_cfg.modules:
        query_cfg.modules = list(load_gradients(query_cfg.query_path).dtype.names)

    query_ds = get_query_data(query_cfg)
    query_ds = query_ds.with_format("torch", columns=query_cfg.modules)

    start = perf_counter()
    attr = Attributor(
        index_cfg.run_path,
        device=device,
        faiss_cfg=FaissConfig("IVF1,SQfp16", mmap_index=True, num_shards=5),
        unit_norm=query_cfg.unit_normalize,
    )
    print(f"Attributor loaded in {perf_counter() - start}")

    # print({name: torch.tensor(query_ds[:][name]) for name in query_cfg.modules})

    print("Searching...")
    start = perf_counter()
    scores, indices = attr.search(
        {
            name: torch.tensor(query_ds[:][name]).to(device)
            for name in query_cfg.modules
        },
        k,
    )

    print(f"Query time: {perf_counter() - start}")

    data = {
        "scores": scores,
        "indices": indices,
    }
    print(data)
    print("Max score", scores.max())
    print("Min score", scores.min())
    print("Mean score", scores.mean())
    print("Std score", scores.std())

    dataset = Dataset.from_dict(data)
    dataset.save_to_disk(Path(query_cfg.query_path) / "trial" / "static_query.hf")

    # Get the text rows associated with the top 50 scores
    text_ds = get_text_rows(indices, scores, index_cfg)
    print(text_ds)

    # Add the scores to the text rows
    dataset.save_to_disk(Path(query_cfg.query_path) / "trial" / "static_query.hf")
