from pathlib import Path
from time import perf_counter

import torch
from datasets import Dataset

from bergson import Attributor, FaissConfig

from .data import (
    IndexConfig,
    QueryConfig,
    load_gradients,
)
from .dynamic_query import get_query_data


@torch.inference_mode()
def query_existing(
    query_cfg: QueryConfig,
    index_cfg: IndexConfig,
    k: int | None,
    device="cpu",
):
    if not query_cfg.modules:
        query_cfg.modules = list(load_gradients(query_cfg.query_path).dtype.names)

    query_ds = get_query_data(query_cfg)
    query_ds = query_ds.with_format("torch", columns=query_cfg.modules)

    start = perf_counter()
    attr = Attributor(
        index_cfg.run_path,
        device=device,
        faiss_cfg=FaissConfig(mmap_index=True, num_shards=5),
        unit_norm=query_cfg.unit_normalize,
    )
    print(f"Attributor loaded in {perf_counter() - start}")

    print("Searching...")
    start = perf_counter()

    search_inputs = {
        name: torch.as_tensor(query_ds[:][name]).to(device)
        for name in query_cfg.modules
    }

    scores, indices = attr.search(search_inputs, k)

    print(f"Query time: {perf_counter() - start}")

    scores_tensor = torch.as_tensor(scores).cpu()
    indices_tensor = torch.as_tensor(indices).to(torch.int64).cpu()

    num_queries, num_scores = scores_tensor.shape
    print(
        f"Collected scores for {num_queries} queries " f"with {num_scores} scores each."
    )

    output_path = Path(query_cfg.scores_path) / "scores.hf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_dict(
        {
            "scores": scores_tensor.tolist(),
            "indices": indices_tensor.tolist(),
        }
    )
    dataset.save_to_disk(output_path)
    print(f"Saved raw search results to {output_path}")
