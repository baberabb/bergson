import tempfile
from pathlib import Path

import torch
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


def test_GPTNeoX():
    temp_dir = Path(tempfile.mkdtemp())
    print(temp_dir)

    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-GPTNeoXForCausalLM")
    model = AutoModelForCausalLM.from_config(config)

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], device=model.device)
    inputs = dict(
        input_ids=tokens,
        labels=tokens,
    )
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

            assert torch.isfinite(g).all()
            assert torch.isfinite(collected_grad.squeeze(0)).all()

            # The test computes A @ weight.grad @ B.T, while GradientCollector computes
            # (G @ A.T).mT @ (I @ B.T), which are mathematically equivalent.
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
                    # different order of operations.
                    assert torch.isfinite(g).all()
                    assert torch.isfinite(collected_grad.squeeze(0)).all()

                    torch.testing.assert_close(
                        g, collected_grad.squeeze(0).view_as(g), atol=1e-4, rtol=1e-4
                    )
                    # Check gradients are the same after loading and restoring
                    if do_load:
                        torch.testing.assert_close(
                            collected_grad, previous_collected_grads[name]
                        )

                previous_collected_grads = collected_grads.copy()
