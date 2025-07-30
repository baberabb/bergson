from simple_parsing import parse

from bergson.data import IndexConfig
from bergson.distributed import distributed_computing
from bergson.gradients import GradientProcessor
from bergson.hessians.ekfac_compute import EkfacApplicator


def ekfac_apply_worker(
    cfg: IndexConfig,
    processor: GradientProcessor,
):
    attributor = EkfacApplicator(
        processor=processor,
        cfg=cfg,
    )

    attributor.prepare_attribution()
    attributor.compute_ivhp_sharded()


if __name__ == "__main__":
    distributed_computing(
        cfg=parse(IndexConfig), worker_fn=ekfac_apply_worker, setup_data=False, setup_model=False, setup_processor=False
    )
