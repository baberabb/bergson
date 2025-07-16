from bergson.distributed import distributed_computing

from .data import IndexConfig
from .processing import collect_gradients


def build_gradient_dataset(cfg: IndexConfig):
    distributed_computing(
        cfg,
        worker_fn=collect_gradients,
    )
