import time
from argparse import ArgumentParser

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientCollector,
    GradientProcessor,
)

torch._dynamo.cache_size = 0  # Disable cache size limit for this example

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size to use for the test",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-1b",
        help="Model to use for the test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=512,
        help="Length of the input sequence",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_config(config)
    model.to("cuda:0")
    model.float()

    tokens = torch.arange(args.sequence_length, device=model.device)
    tokens = tokens.unsqueeze(0).expand(args.batch_size, -1)
    inputs = dict(input_ids=tokens, labels=tokens)

    print("Warming up the torch.compile cache...")
    model(**inputs).loss.backward()
    model.zero_grad()

    print("Running with no hook at all")
    torch.cuda.synchronize()
    start = time.monotonic()

    model(**inputs).loss.backward()

    torch.cuda.synchronize()
    end = time.monotonic()
    model.zero_grad()
    print(f"Time taken: {end - start:.3f} seconds")

    print("Running with no normalization")
    torch.cuda.synchronize()
    start = time.monotonic()

    with GradientCollector(model) as collector:
        model(**inputs).loss.backward()

    torch.cuda.synchronize()
    end = time.monotonic()
    print(f"Time taken: {end - start:.3f} seconds")

    adafactors: dict[str, AdafactorNormalizer] = {}
    adams: dict[str, AdamNormalizer] = {}

    p = collector.processor.projection_dim
    assert p is not None, "Projection dimension must be set"

    # Go through the motions of what GradientCollector does, but after the fact
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue

        o, i = layer.out_features, layer.in_features
        A = collector.projection(name, p, o, "left")
        B = collector.projection(name, p, i, "right")

        g = layer.weight.grad
        assert g is not None

        # Store normalizers for this layer
        moments = g.square()
        adams[name] = AdamNormalizer(moments)
        adafactors[name] = adams[name].to_adafactor()

    # Now do it again but this time use the normalizers
    for normalizers in (adafactors, adams):
        model.zero_grad()

        print(
            f"Using {type(next(iter(normalizers.values()))).__name__} for normalization"
        )
        torch.cuda.synchronize()
        start = time.monotonic()

        with GradientCollector(model, GradientProcessor(normalizers)) as collector:
            model(**inputs).loss.backward()

        torch.cuda.synchronize()
        end = time.monotonic()
        print(f"Time taken: {end - start:.3f} seconds")
