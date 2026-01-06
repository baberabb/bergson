import json
from pathlib import Path
from dataclasses import asdict

from datasets import Dataset, load_dataset
from simple_parsing import ArgumentParser, ConflictResolution
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import Attributor, FaissConfig
from bergson.config import QueryConfig, IndexConfig
from bergson.utils.utils import assert_type

from bergson.utils.worker_utils import setup_model_and_peft



def query(query_cfg: QueryConfig):
    """
    Run an interactive CLI session that queries a pre-built gradient index.

    Parameters
    ----------
    cfg : QueryConfig
        Configuration describing the index path, HF model to load, and dataset field
        used to print the retrieved documents.
    """
    with open(Path(query_cfg.index) / "index_config.json", "r") as f:
        index_cfg = json.load(f)

    dataset_name = index_cfg["data"]["dataset"]
    if not query_cfg.model:
        query_cfg.model = index_cfg["model"]

    # Support loading a different model than the one the index was built for, e.g.
    # a different checkpoint.
    if query_cfg.model:
        query_index_cfg = IndexConfig(
            **{k: v for k, v in index_cfg.items() if k != 'model'},
            model=query_cfg.model,
        )
        tokenizer = AutoTokenizer.from_pretrained(query_cfg.model)
        model, _ = setup_model_and_peft(query_index_cfg, 0)
    else:
        tokenizer = AutoTokenizer.from_pretrained(index_cfg.model)
        model, _ = setup_model_and_peft(index_cfg, 0)

    dataset = load_dataset(dataset_name, split="train")
    dataset = assert_type(Dataset, dataset)

    faiss_cfg = FaissConfig() if query_cfg.faiss else None
    attr = Attributor(Path(query_cfg.index), device="cuda", faiss_cfg=faiss_cfg)

    # Query loop
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break

        # Tokenize the query
        inputs = tokenizer(query, return_tensors="pt").to("cuda:0")
        x = inputs["input_ids"]

        with attr.trace(model.base_model, 5) as result:
            model(x, labels=x).loss.backward()
            model.zero_grad()

        # Print the results
        print(f"Top 5 results for '{query}':")
        for i, (d, idx) in enumerate(
            zip(result.scores.squeeze(), result.indices.squeeze())
        ):
            if idx.item() == -1:
                print("Found invalid result, skipping")
                continue

            text = dataset[int(idx.item())][query_cfg.text_field]
            print(text[:5000])

            print(f"{i + 1}: (distance: {d.item():.4f})")


def main():
    """Parse arguments for `query_index.py` and launch the REPL."""
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(QueryConfig, dest="prog")
    prog: QueryConfig = parser.parse_args().prog
    query(prog)


if __name__ == "__main__":
    main()
