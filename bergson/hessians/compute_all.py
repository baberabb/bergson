import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedModel

from bergson.data import IndexConfig
from bergson.gradients import GradientProcessor
from bergson.hessians.ekfac_compute import EkfacComputer


def compute_all_factors(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    *,
    batches: list[list[int]],
    target_modules: set[str] | None = None,
    cfg: IndexConfig,
):
    computer = EkfacComputer(
        model=model,
        processor=processor,
        data=data,
        batches=batches,
        target_modules=target_modules,
        cfg=cfg,
    )
    computer.compute_covariance()

    dist.barrier() if dist.is_initialized() else None

    computer.compute_eigendecomposition(covariance_type="activation")
    computer.compute_eigendecomposition(covariance_type="gradient")

    dist.barrier() if dist.is_initialized() else None
    computer.compute_eigenvalue_correction()
