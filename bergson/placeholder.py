from datasets import Dataset
from transformers import PreTrainedModel

from bergson.collector.collector import CollectorComputer, GradientCollector
from bergson.config import AttentionConfig, IndexConfig, ReduceConfig
from bergson.gradients import GradientProcessor
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
