from argparse import ArgumentParser

from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson import Attributor


def main():
    parser = ArgumentParser()
    parser.add_argument("index", type=str)
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": "cuda:0"})

    attr = Attributor(args.index, device="cuda:0")

    # Query loop
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break

        # Tokenize the query
        inputs = tokenizer(query, return_tensors="pt").to("cuda:0")
        x = inputs["input_ids"]

        with attr.trace(model, 5) as result:
            model(x, labels=x).loss.backward()
            model.zero_grad()

        # Print the results
        print(f"Top 5 results for '{query}':")
        for i, (d, idx) in enumerate(zip(result.scores, result.indices)):
            # tokens = dataset[idx.item()]["input_ids"]
            # string = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"{i + 1}: (distance: {d.item():.4f})")


if __name__ == "__main__":
    main()
