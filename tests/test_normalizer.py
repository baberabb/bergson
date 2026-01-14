import torch.nn as nn

from bergson import fit_normalizers
from bergson.config import IndexConfig


def test_fit_normalizers_runs(tmp_path, model, dataset):
    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, nn.Linear)
    }
    print("len dataset", len(dataset))
    print("target_modules", target_modules)

    dataset = dataset.repeat(10)

    normalizers = fit_normalizers(
        model,
        dataset,
        cfg=IndexConfig(
            run_path=str(tmp_path),
            skip_preconditioners=True,
            normalizer="adam",
        ),
        batches=[[idx] for idx in range(len(dataset))],
        target_modules=target_modules,
    )

    assert len(normalizers) == len(target_modules)
