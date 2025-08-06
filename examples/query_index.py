from argparse import ArgumentParser

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import Attributor


def main():
    parser = ArgumentParser()
    parser.add_argument("index", type=str)
    parser.add_argument(
        "--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    parser.add_argument("--dataset", type=str, default="EleutherAI/SmolLM2-135M-10B")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--unit_norm", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": "cuda:0"})
    dataset = load_dataset(args.dataset, split="train")

    attr = Attributor(args.index, device="cuda:0")

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

            text = dataset[idx.item()][args.text_field]
            print(text[:5000])

            print(f"{i + 1}: (distance: {d.item():.4f})")


if __name__ == "__main__":
    main()
