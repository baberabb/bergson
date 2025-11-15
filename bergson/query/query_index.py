import json
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, load_dataset
from simple_parsing import ArgumentParser, ConflictResolution
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import Attributor, FaissConfig
from bergson.utils import assert_type


@dataclass
class QueryConfig:
    index: str = ""
    """Path to the existing index."""

    model: str = ""
    """Model to use for the query. When not provided the model used to build the
    index is used."""

    text_field: str = "text"
    """Field to use for the query."""

    unit_norm: bool = False
    """Whether to unit normalize the query."""

    faiss: bool = False
    """Whether to use FAISS for the query."""


def query(cfg: QueryConfig):
    with open(Path(cfg.index) / "index_config.json", "r") as f:
        index_cfg = json.load(f)

    dataset_name = index_cfg["data"]["dataset"]
    if not cfg.model:
        cfg.model = index_cfg["model"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, device_map={"": "cuda:0"})
    dataset = load_dataset(dataset_name, split="train")
    dataset = assert_type(Dataset, dataset)

    faiss_cfg = FaissConfig() if cfg.faiss else None
    attr = Attributor(Path(cfg.index), device="cuda", faiss_cfg=faiss_cfg)

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

            text = dataset[int(idx.item())][cfg.text_field]
            print(text[:5000])

            print(f"{i + 1}: (distance: {d.item():.4f})")


def main():
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(QueryConfig, dest="prog")
    prog: QueryConfig = parser.parse_args().prog
    query(prog)


if __name__ == "__main__":
    main()
