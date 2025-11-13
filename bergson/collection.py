import torch
from datasets import Dataset
from transformers import PreTrainedModel

from bergson.collector.collector import CollectorComputer
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import AttentionConfig, IndexConfig, ReduceConfig
from bergson.gradients import GradientProcessor
from bergson.score.scorer import Scorer


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
    Compute gradients using the hooks specified in the GradientCollector.
    """
    if attention_cfgs is None:
        attention_cfgs = {}
    collector = GradientCollector(
        model=model.base_model,  # type: ignore
        cfg=cfg,
        processor=processor,
        target_modules=target_modules,
        data=data,
        scorer=scorer,
        reduce_cfg=reduce_cfg,
        attention_cfgs=attention_cfgs or {},
    )

    validate_batch_size(model, cfg.token_batch_size, collector)

    computer = CollectorComputer(
        model=model,  # type: ignore
        data=data,
        collector=collector,
        batches=batches,
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="New worker - Collecting gradients")


def validate_batch_size(
    model: PreTrainedModel,
    token_batch_size: int | None,
    collector: GradientCollector,
):
    """Validate that the specified token batch size fits on device."""
    if token_batch_size is None:
        return

    # Check that token_batch_size doesn't exceed model's max sequence length
    max_seq_len = getattr(model.config, "max_position_embeddings", None)
    if max_seq_len is not None and token_batch_size > max_seq_len:
        raise ValueError(
            f"Token batch size {token_batch_size} exceeds model's max sequence length "
            f"({max_seq_len}). Use --token_batch_size {max_seq_len} or smaller."
        )

    random_tokens = torch.randint(
        0, 10, (1, token_batch_size), device=model.device, dtype=torch.long
    )
    try:
        with collector:
            loss = model(random_tokens).logits[0, 0, 0].float()
            loss.backward()
            model.zero_grad()
    except Exception as e:
        raise ValueError(
            f"Token batch size {token_batch_size} is too large for the device. "
            f"Try reducing the batch size or use --fsdp to shard the model."
        ) from e
