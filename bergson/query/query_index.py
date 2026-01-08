import json
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from bergson import Attributor, FaissConfig
from bergson.config import IndexConfig, QueryConfig
from bergson.data import load_data_string
from bergson.utils.utils import setup_reproducibility
from bergson.utils.worker_utils import setup_model_and_peft


def query(
    query_cfg: QueryConfig,
):
    """
    Run an interactive CLI session that queries a pre-built gradient index.

    Parameters
    ----------
    cfg : QueryConfig
        Configuration describing the index path, HF model to load, and dataset field
        used to print the retrieved documents.
    """
    with open(Path(query_cfg.index) / "index_config.json", "r") as f:
        index_cfg = IndexConfig(**json.load(f))

    if index_cfg.debug:
        setup_reproducibility()

    # Load a different model than the one the index was built for, e.g.
    # a different checkpoint.
    if query_cfg.model:
        query_index_cfg = IndexConfig(
            **{k: v for k, v in asdict(index_cfg).items() if k != "model"},
            model=query_cfg.model,
        )
        tokenizer = AutoTokenizer.from_pretrained(query_cfg.model)
        model, target_modules = setup_model_and_peft(
            query_index_cfg, 0, device_map_auto=query_cfg.device_map_auto
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(index_cfg.model)
        model, target_modules = setup_model_and_peft(
            index_cfg, 0, device_map_auto=query_cfg.device_map_auto
        )

    ds = load_data_string(
        index_cfg.data.dataset,
        index_cfg.data.split,
        index_cfg.data.subset,
        index_cfg.data.data_args,
    )

    faiss_cfg = FaissConfig() if query_cfg.faiss else None
    attr = Attributor(Path(query_cfg.index), device="cuda", faiss_cfg=faiss_cfg)

    # Get the device of the first model parameter for multi-GPU setups
    model_device = next(model.parameters()).device

    # Query loop
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break

        # Tokenize the query
        inputs = tokenizer(query, return_tensors="pt").to(model_device)
        x = inputs["input_ids"]

        with attr.trace(
            model.base_model, 5, modules=target_modules, reverse=query_cfg.reverse
        ) as result:
            model(x, labels=x).loss.backward()
            model.zero_grad()

        # Print the results
        mode = "Bottom" if query_cfg.reverse else "Top"
        print(f"{mode} 5 results for '{query}':")
        for i, (d, idx) in enumerate(
            zip(result.scores.squeeze(), result.indices.squeeze())
        ):
            if idx.item() == -1:
                print("Found invalid result, skipping")
                continue

            text = str(ds[int(idx.item())][query_cfg.text_field])  # type: ignore[arg-type]
            print(text[:2000])
            if len(text) > 2000:
                print(". . .")

            print(f"{i + 1}: (distance: {d.item():.4f})")
