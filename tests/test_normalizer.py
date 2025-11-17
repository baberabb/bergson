import torch.nn as nn

from bergson import fit_normalizers


def test_fit_normalizers_runs(model, dataset):
    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, nn.Linear)
    }
    normalizers = fit_normalizers(
        model,
        dataset,
        batches=[[idx] for idx in range(len(dataset))],
        kind="adam",
        target_modules=target_modules,
    )

    assert len(normalizers) == len(target_modules)
