import tempfile
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig
from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientProcessor,
    LayerAdapter,
)


def test_gradient_collector_proj_norm():
    temp_dir = Path(tempfile.mkdtemp())
    print(temp_dir)

    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-GPTNeoXForCausalLM")
    model = AutoModelForCausalLM.from_config(config)

    # It's important that we use a batch size of one so that we can simply use the
    # aggregate gradients from the backward itself and compare against those
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=model.device)
    inputs = dict(input_ids=tokens, labels=tokens)
    data = Dataset.from_dict({"input_ids": tokens.tolist()})

    # Test with 16 x 16 random projection as well as with no projection
    for p in (16, None):
        cfg = IndexConfig(
            run_path=str(temp_dir / "run"),
            skip_index=True,
            skip_preconditioners=p is None,
        )
        processor = GradientProcessor(projection_dim=p)
        collector = GradientCollector(
            model=model,
            cfg=cfg,
            data=data,
            processor=processor,
        )
        with collector:
            model.zero_grad()
            model(**inputs).loss.backward()
            collected_grads = collector.mod_grads.copy()

        adafactors: dict[str, AdafactorNormalizer] = {}
        adams: dict[str, AdamNormalizer] = {}

        # Go through the motions of what GradientCollector does, but after the fact
        for name, collected_grad in collected_grads.items():
            layer = model.get_submodule(name)

            i = getattr(layer, LayerAdapter.in_attr(layer))
            o = getattr(layer, LayerAdapter.out_attr(layer))

            g = layer.weight.grad
            assert g is not None

            moments = g.square()

            if p is not None:
                A = collector.projection(name, p, o, "left", g.device, g.dtype)
                B = collector.projection(name, p, i, "right", g.device, g.dtype)
                g = A @ g @ B.T

            torch.testing.assert_close(g, collected_grad.squeeze(0).view_as(g))

            # Store normalizers for this layer
            adams[name] = AdamNormalizer(moments)
            adafactors[name] = adams[name].to_adafactor()

        # Now do it again but this time use the normalizers
        for normalizers in (adams, adafactors):
            previous_collected_grads = {}
            for do_load in (False, True):
                if do_load:
                    processor = GradientProcessor.load(temp_dir / "processor")
                else:
                    processor = GradientProcessor(
                        normalizers=normalizers, projection_dim=p
                    )
                    processor.save(temp_dir / "processor")

                collector.processor = processor
                with collector:
                    model.zero_grad()
                    model(**inputs).loss.backward()
                    collected_grads = collector.mod_grads.copy()

                for name, collected_grad in collected_grads.items():
                    layer = model.get_submodule(name)
                    i = getattr(layer, LayerAdapter.in_attr(layer))
                    o = getattr(layer, LayerAdapter.out_attr(layer))
                    g = layer.weight.grad
                    assert g is not None

                    g = normalizers[name].normalize_(g)
                    if p is not None:
                        A = collector.projection(name, p, o, "left", g.device, g.dtype)
                        B = collector.projection(name, p, i, "right", g.device, g.dtype)
                        g = A @ g @ B.T

                    # Compare the normalized gradient with the collected gradient. We
                    # use a higher tolerance than the default because there seems to be
                    # some non-negligible numerical error that accumulates due to the
                    # different order of operations. Maybe we should look into this
                    torch.testing.assert_close(
                        g, collected_grad.squeeze(0).view_as(g), atol=1e-4, rtol=1e-4
                    )
                    # Check gradients are the same after loading and restoring
                    if do_load:
                        torch.testing.assert_close(
                            collected_grad, previous_collected_grads[name]
                        )

                previous_collected_grads = collected_grads.copy()


@pytest.mark.parametrize("include_bias", [True, False])
def test_gradient_collector_batched(include_bias: bool):
    torch.manual_seed(42)
    N = 4
    S = 6
    I = 5
    O = 3

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(I, O * 2, bias=include_bias)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(O * 2, O, bias=include_bias)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    torch.manual_seed(42)
    model = SimpleModel()

    optimizer = torch.optim.Adam(model.parameters())

    # Run a few training steps to build up second moments
    for _ in range(5):
        optimizer.zero_grad()
        out = model(torch.randn(N, S, I))
        loss = (out**2).sum()
        loss.backward()
        optimizer.step()

    normalizers = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            layer_name = name.replace(".weight", "")
            # Adam stores second moments as 'exp_avg_sq'
            exp_avg_sq = optimizer.state[param]["exp_avg_sq"]
            normalizers[layer_name] = AdamNormalizer(exp_avg_sq)

    # collect gradients
    collected_grads = {}

    def closure(name: str, g: torch.Tensor):
        """Store the gradients in a dictionary for later comparison."""
        collected_grads[name] = g

    processor = GradientProcessor(
        normalizers=normalizers, projection_dim=None, include_bias=include_bias
    )
    collector = GradientCollector(model, closure, processor)

    x = torch.randn(N, S, I)
    with collector:
        model.zero_grad()
        out = model(x)
        loss = (out**2).sum()
        loss.backward()

    def compute_ground_truth():
        """Compute gradients using individual backward passes, with normalization."""
        model.zero_grad()
        output = model(x)  # [N, S, O]

        # Per-sample losses
        per_sample_losses = (output**2).sum(dim=(1, 2))  # [N]

        ground_truth_grads = defaultdict(list)
        for n in range(N):
            model.zero_grad()
            per_sample_losses[n].backward(retain_graph=True)

            # manually normalize
            for layer_name in ["fc1", "fc2"]:
                layer = model.get_submodule(layer_name)
                grad = layer.weight.grad.clone()

                grad = normalizers[layer_name].normalize_(grad)

                if include_bias:
                    bias_grad = layer.bias.grad.clone()
                    bias_grad = bias_grad.unsqueeze(1)
                    grad = torch.cat([grad, bias_grad], dim=1)

                ground_truth_grads[layer_name].append(grad)

        for layer_name in ["fc1", "fc2"]:
            ground_truth_grads[layer_name] = torch.stack(ground_truth_grads[layer_name])

        return ground_truth_grads

    ground_truth = compute_ground_truth()
    for layer_name in ["fc1", "fc2"]:
        torch.testing.assert_close(
            collected_grads[layer_name], ground_truth[layer_name]
        )


def test_bias_gradients():
    """Test that per-sample bias gradients are correctly computed."""
    torch.manual_seed(42)
    N = 4
    S = 6
    I = 5
    O = 3

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(I, O, bias=True)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    x = torch.randn(N, S, I)

    # bias gradient is a sum over sequence dimension for each n
    def compute_ground_truth(model) -> torch.Tensor:
        """Compute gradients using individual backward passes."""
        model.zero_grad()
        output = model(x)  # [N, S, O]

        per_sample_losses = (output**2).sum(dim=(1, 2))  # [N]

        bias_grads = []
        for n in range(N):
            model.zero_grad()
            per_sample_losses[n].backward(retain_graph=True)
            bias_grads.append(model.fc.bias.grad.clone())

        return torch.stack(bias_grads, dim=0)  # [N, O]

    ground_truth = compute_ground_truth(model)

    # GradientCollector with include_bias=True
    collected_grads = {}

    def closure(name: str, g: torch.Tensor):
        collected_grads[name] = g

    processor = GradientProcessor(include_bias=True, projection_dim=None)
    collector = GradientCollector(model, closure, processor, target_modules={"fc"})

    with collector:
        model.zero_grad()
        output = model(x)
        loss = (output**2).sum()
        loss.backward()

    # the last column is bias
    bias_grads = collected_grads["fc"][..., -1]

    assert bias_grads.shape == (
        N,
        3,
    ), f"Expected shape ({N}, {O}), got {bias_grads.shape}"
    assert ground_truth.shape == (
        N,
        3,
    ), f"Expected shape ({N}, {O}), got {ground_truth.shape}"

    # Compare to ground truth
    torch.testing.assert_close(bias_grads, ground_truth)
