import os
import socket
from typing import Any, Callable

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes

from .config import DistributedConfig


def dist_worker(
    worker: Callable,
    *worker_args,
):
    try:
        worker(*worker_args)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                print(f"Barrier failed during cleanup: {e}")
                pass

            dist.destroy_process_group()


def launch_distributed_run(
    process_name: str,
    worker,
    const_worker_args: list[Any],
    dist_config: DistributedConfig | None = None,
):
    if dist_config is None:
        dist_config = DistributedConfig()

    local_world_size = dist_config.nproc_per_node
    world_size = dist_config.world_size
    start_rank = dist_config.start_rank

    # Multi-node environment
    if dist_config.nnode > 1:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
    else:
        master_addr = "localhost"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, master_port = s.getsockname()
        master_port = str(master_port)

    if world_size <= 1:
        worker(0, 0, 1, *const_worker_args)
    else:
        mp.set_sharing_strategy("file_system")

        ctx = None
        try:
            ctx = start_processes(
                process_name,
                dist_worker,
                args={
                    i: (worker, start_rank + i, i, world_size, *const_worker_args)
                    for i in range(local_world_size)
                },
                envs={
                    i: {
                        "LOCAL_RANK": str(i),
                        "RANK": str(start_rank + i),
                        "WORLD_SIZE": str(world_size),
                        "MASTER_ADDR": master_addr,
                        "MASTER_PORT": master_port,
                    }
                    for i in range(local_world_size)
                },
                logs_specs=DefaultLogsSpecs(),
            )
            ctx.wait()
        finally:
            if ctx is not None:
                ctx.close()  # Kill any processes that are still running
