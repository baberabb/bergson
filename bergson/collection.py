import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedModel

from bergson.collector.collector import CollectorComputer, GradientCollector
from bergson.config import AttentionConfig, IndexConfig, ReduceConfig
from bergson.gradients import GradientProcessor

# from bergson.placeholder import collect_gradients_new
from bergson.score.scorer import Scorer


def collect_gradients_new(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    cfg: IndexConfig,
    *,
    batches: list[list[int]] | None = None,
    target_modules: set[str] | None = None,
    attention_cfgs: dict[str, AttentionConfig] = {},
    scorer: Scorer | None = None,
    reduce_cfg: ReduceConfig | None = None,
):
    collector = GradientCollector(
        model=model.base_model,  # type: ignore
        cfg=cfg,
        processor=processor,
        target_modules=target_modules,
        data=data,
        attention_cfgs=attention_cfgs,
        scorer=scorer,
        reduce_cfg=reduce_cfg,
    )

    computer = CollectorComputer(
        model=model,  # type: ignore
        data=data,
        collector=collector,
        batches=batches,
        cfg=cfg,
    )
    computer._compute(desc="New worker - Collecting gradients")


def collect_gradients(
    model: PreTrainedModel,
    data: Dataset,
    processor: GradientProcessor,
    cfg: IndexConfig,
    *,
    batches: list[list[int]] | None = None,
    target_modules: set[str] | None = None,
    attention_cfgs: dict[str, AttentionConfig] | None = None,
    scorer: Scorer | None = None,
    reduce_cfg: ReduceConfig | None = None,
):
    """
    Compute projected gradients using a subset of the dataset.
    """
    if True:
        collect_gradients_new(
            model,
            data,
            processor,
            cfg,
            batches=batches,
            target_modules=target_modules,
            attention_cfgs=attention_cfgs or {},
            scorer=scorer,
            reduce_cfg=reduce_cfg,
        )
        return
    # else:
    #     preconditioners = processor.preconditioners

    #     def callback(name: str, g: torch.Tensor):
    #         # Compute the outer product of the flattened gradient
    #         if not cfg.skip_preconditioners:
    #             g = g.float()
    #             preconditioner = preconditioners.get(name, None)
    #             if preconditioner is None:
    #                 preconditioners[name] = g.mT @ g
    #             else:
    #                 preconditioner.addmm_(g.mT, g)

    #     for indices in tqdm(batches, disable=rank != 0, desc="Building index"):
    #         if cfg.loss_fn == "kl":
    #             with torch.inference_mode():
    #                 set_peft_enabled(model, False)
    #                 ref_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)
    #                 set_peft_enabled(model, True)

    #             with collector:
    #                 ft_lps = torch.log_softmax(model(x).logits[:, :-1], dim=-1)

    #                 # Compute average KL across all unmasked tokens
    #                 kls = torch.sum(ft_lps.exp() * (ft_lps - ref_lps), dim=-1)
    #                 losses = torch.sum(kls * masks, dim=-1) / denoms

    #                 losses.mean().backward()

    #     process_preconditioners(processor, preconditioners, len(data))
    #     return


def process_preconditioners(
    processor: GradientProcessor,
    preconditioners: dict[str, torch.Tensor],
    len_data: int,
):
    """
    Aggregate preconditioners across ranks and compute their eigen decomposition
    distributed across all ranks.
    """

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    preconditioners_eigen = {}
    if rank == 0:
        print("Saving preconditioners...")
    for name, prec in preconditioners.items():
        if dist.is_initialized():
            dist.all_reduce(prec)

        preconditioners[name] = prec / len_data

    processor.preconditioners = preconditioners

    if rank == 0:
        print("Computing preconditioner eigen decompositions...")
    names = list(preconditioners.keys())
    names_per_rank = names[rank::world_size]

    for name in names_per_rank:
        original_dtype = preconditioners[name].dtype
        prec = preconditioners[name].to(dtype=torch.float64)
        eigvals, eigvecs = torch.linalg.eigh(prec)
        preconditioners_eigen[name] = (
            eigvals.to(dtype=original_dtype).contiguous(),
            eigvecs.to(dtype=original_dtype).contiguous(),
        )

    if rank == 0:
        print("Gathering and saving preconditioner eigen decompositions...")

    for name in names:
        prec = preconditioners[name]
        if name not in preconditioners_eigen:
            eigval = torch.zeros(prec.size(0), dtype=prec.dtype, device=prec.device)
            eigvec = torch.zeros_like(prec)
        else:
            eigval, eigvec = preconditioners_eigen[name]

        dist.all_reduce(eigval, op=dist.ReduceOp.SUM) if dist.is_initialized() else None
        dist.all_reduce(eigvec, op=dist.ReduceOp.SUM) if dist.is_initialized() else None

        preconditioners_eigen[name] = (eigval, eigvec)
    if rank == 0:
        processor.preconditioners_eigen = preconditioners_eigen
