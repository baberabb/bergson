import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.gradients import (
    AdafactorNormalizer,
    AdamNormalizer,
    GradientCollector,
    GradientProcessor,
    LayerAdapter,
)


def test_phi3():
    temp_dir = Path(tempfile.mkdtemp())

    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(config)

    # It's important that we use a batch size of one so that we can simply use the
    # aggregate gradients from the backward itself and compare against those
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=model.device)
    inputs = dict(input_ids=tokens, labels=tokens)
    collected_grads = {}

    def closure(name: str, g: torch.Tensor):
        """Store the gradients in a dictionary for later comparison."""
        collected_grads[name] = g

    # Test with 16 x 16 random projection as well as with no projection
    for p in (16, None):
        processor = GradientProcessor(projection_dim=p)
        collector = GradientCollector(model, closure, processor)
        with collector:
            model.zero_grad()
            model(**inputs).loss.backward()

        adafactors: dict[str, AdafactorNormalizer] = {}
        adams: dict[str, AdamNormalizer] = {}

        # Go through the motions of what GradientCollector does, but after the fact
        for name, collected_grad in collected_grads.items():
            layer = model.get_submodule(name)

            i = getattr(layer, LayerAdapter.in_attr(layer))
            o = getattr(layer, LayerAdapter.out_attr(layer))

            weight_grad = layer.weight.grad
            assert weight_grad is not None
            bias_grad = layer.bias.grad if getattr(layer, "bias", None) is not None else None

            moments = weight_grad.square()
            g = weight_grad
            if bias_grad is not None:
                g = torch.cat([g, bias_grad.unsqueeze(-1)], dim=-1)
            if p is not None:
                A = collector.projection(name, p, o, "left", g.device, g.dtype)
                right_dim = i + (1 if bias_grad is not None else 0)
                B = collector.projection(name, p, right_dim, "right", g.device, g.dtype)
                g = A @ g @ B.T

            torch.testing.assert_close(g, collected_grad.squeeze(0))

            # Store normalizers for this layer
            adams[name] = AdamNormalizer(moments)
            adafactors[name] = adams[name].to_adafactor()

        # Now do it again but this time use the normalizers
        for normalizers in (adafactors, adams):
            previous_collected_grads = {}
            for do_load in (False, True):
                if do_load:
                    processor = GradientProcessor.load(str(temp_dir / "processor"))
                else:
                    processor = GradientProcessor(
                        normalizers=normalizers, projection_dim=p
                    )
                    processor.save(str(temp_dir / "processor"))
                collector = GradientCollector(model, closure, processor)
                with collector:
                    model.zero_grad()
                    model(**inputs).loss.backward()

                for name, collected_grad in collected_grads.items():
                    layer = model.get_submodule(name)
                    i = getattr(layer, LayerAdapter.in_attr(layer))
                    o = getattr(layer, LayerAdapter.out_attr(layer))
                    weight_grad = layer.weight.grad
                    assert weight_grad is not None
                    bias_grad = (
                        layer.bias.grad if getattr(layer, "bias", None) is not None else None
                    )

                    weight_grad = normalizers[name].normalize_(weight_grad.clone())
                    if bias_grad is not None:
                        if isinstance(normalizers[name], AdafactorNormalizer):
                            r = normalizers[name].row.add(1e-30)
                            row_scale = r.mean().sqrt() * r.rsqrt()
                            bias_component = bias_grad * row_scale.type_as(bias_grad)
                        else:
                            bias_component = bias_grad
                        g = torch.cat([weight_grad, bias_component.unsqueeze(-1)], dim=-1)
                    else:
                        g = weight_grad
                    if p is not None:
                        A = collector.projection(name, p, o, "left", g.device, g.dtype)
                        right_dim = i + (1 if bias_grad is not None else 0)
                        B = collector.projection(name, p, right_dim, "right", g.device, g.dtype)
                        g = A @ g @ B.T

                    # Compare the normalized gradient with the collected gradient. We
                    # use a higher tolerance than the default because there seems to be
                    # some non-negligible numerical error that accumulates due to the
                    # different order of operations. Maybe we should look into this
                    torch.testing.assert_close(
                        g, collected_grad.squeeze(0), atol=1e-4, rtol=1e-4
                    )
                    # Check gradients are the same after loading and restoring
                    if do_load:
                        torch.testing.assert_close(
                            collected_grad, previous_collected_grads[name]
                        )

                previous_collected_grads = collected_grads.copy()
