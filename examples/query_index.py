from argparse import ArgumentParser

from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import GradientCollector, GradientProcessor
from bergson.data import load_index


def main():
    parser = ArgumentParser()
    parser.add_argument("index", type=str)
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": "cuda:0"})

    processor = GradientProcessor.load(args.index, map_location="cuda:0")
    dataset = load_index(args.index).with_format("torch")

    print("Loading gradients from disk...")
    grads = dataset["gradient"]
    grads /= grads.norm(dim=1, keepdim=True)  # Normalize the gradients
    print(f"Loaded {len(grads)} gradients.")

    # Query loop
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break

        # Tokenize the query
        inputs = tokenizer(query, return_tensors="pt").to("cuda:0")
        x = inputs["input_ids"]

        with GradientCollector(model, processor) as collector:
            model(x, labels=x).loss.backward()
            model.zero_grad()

        # Pearform the search
        g = collector.flattened_grads()
        prods = grads @ g.cpu().mT
        dists, indices = prods.topk(5, dim=0)

        # Print the results
        print(f"Top 5 results for '{query}':")
        for i, (d, idx) in enumerate(zip(dists, indices)):
            tokens = dataset[idx.item()]["input_ids"]
            string = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"{i + 1}: {string} (distance: {d.item():.4f})")


if __name__ == "__main__":
    main()
