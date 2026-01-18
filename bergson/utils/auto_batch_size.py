"""
In-memory auto batch size determination for Bergson benchmarks.

This module provides utilities to automatically find the optimal token_batch_size
that fits in GPU memory for already-loaded models and datasets.

Main function: find_optimal_token_batch_size()
- Call this with your loaded model, tokenizer, and dataset
- Returns optimal token_batch_size that fits in memory

Adapted from HuggingFace Accelerate's find_executable_batch_size utility.
"""

import gc
import json
from pathlib import Path
from typing import Callable, Optional

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from bergson.collector.collector import CollectorComputer
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import DataConfig, IndexConfig
from bergson.data import allocate_batches, tokenize
from bergson.gradients import GradientProcessor


def should_reduce_batch_size(exception: Exception) -> bool:
    """Check if exception relates to out-of-memory errors or batch size issues."""
    _statements = [
        " out of memory.",
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
        "DefaultCPUAllocator: can't allocate memory",
        "FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed",
        # Catches "Token batch size X exceeds model's max sequence length"
        "Token batch size",
        # Catches "distributed worker error or insufficient documents"
        "insufficient documents",
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def clear_device_cache(garbage_collection: bool = False) -> None:
    """Clear device cache and optionally run garbage collection."""
    if garbage_collection:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def round_to_power_of_2(n: int) -> int:
    """Round down to the nearest power of 2."""
    if n <= 0:
        return 1
    power = 1
    while power * 2 <= n:
        power *= 2
    return power


def save_batch_size_cache(
    cache_path: Path, model_name: str, token_batch_size: int, fsdp: bool = False
) -> None:
    """Save optimal token_batch_size to cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_data = {
        "model_name": model_name,
        "token_batch_size": token_batch_size,
        "fsdp": fsdp,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "gpu_memory_gb": (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available()
            else None
        ),
    }

    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)

    print(f"Saved batch size cache to {cache_path}")


def load_batch_size_cache(
    cache_path: Path, model_name: str, fsdp: bool = False
) -> Optional[int]:
    """Load optimal token_batch_size from cache if available and valid."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        # Verify cache is for the same configuration
        if cache_data.get("model_name") != model_name:
            print(
                f"Cache model mismatch: {cache_data.get('model_name')} != {model_name}"
            )
            return None

        if cache_data.get("fsdp") != fsdp:
            print(f"Cache FSDP mismatch: {cache_data.get('fsdp')} != {fsdp}")
            return None

        # Check if GPU matches (optional warning)
        if torch.cuda.is_available():
            current_gpu = torch.cuda.get_device_name(0)
            cached_gpu = cache_data.get("gpu_name")
            if cached_gpu and cached_gpu != current_gpu:
                print(f"Warning: GPU changed from {cached_gpu} to {current_gpu}")
                print("Cached batch size may not be optimal for this GPU")

        token_batch_size = cache_data.get("token_batch_size")
        if token_batch_size and isinstance(token_batch_size, int):
            print(
                f"Loaded cached token_batch_size={token_batch_size} from {cache_path}"
            )
            return token_batch_size

        return None

    except Exception as e:
        print(f"Failed to load batch size cache: {e}")
        return None


def find_optimal_token_batch_size_raw(
    test_fn: Callable[[int], None],
    starting_batch_size: int = 4096,
    round_to_pow2: bool = True,
    max_batch_size: Optional[int] = None,
) -> int:
    """
    Find optimal token_batch_size by testing with progressively larger/smaller sizes.

    Args:
        test_fn: Function that takes token_batch_size and performs a test pass
        starting_batch_size: Initial token_batch_size to try
        round_to_pow2: Round final batch size down to nearest power of 2
        max_batch_size: Maximum batch size to try (e.g., model's max sequence length)

    Returns:
        Optimal token_batch_size that fits in memory
    """
    token_batch_size = starting_batch_size
    successful_batch_size = None
    iteration = 0

    clear_device_cache(garbage_collection=True)

    while True:
        iteration += 1
        if token_batch_size < 128:
            if successful_batch_size is not None:
                break
            raise RuntimeError(
                f"No executable token_batch_size found, reached minimum (128). "
                f"Started from {starting_batch_size}."
            )

        try:
            print(
                f"[Iteration {iteration}] Trying token_batch_size={token_batch_size}..."
            )
            test_fn(token_batch_size)
            successful_batch_size = token_batch_size
            print(
                f"✓ [Iteration {iteration}] "
                f"token_batch_size={token_batch_size} succeeded"
            )

            # Try larger batch size
            next_size = int(token_batch_size * 1.5)
            # Cap at model's max sequence length if specified
            if max_batch_size is not None:
                next_size = min(next_size, max_batch_size)
            # Cap at a reasonable max (1M tokens) to avoid infinite growth
            if next_size > token_batch_size and token_batch_size < 1_000_000:
                token_batch_size = next_size
                clear_device_cache(garbage_collection=True)
                continue
            else:
                break

        except Exception as e:
            if should_reduce_batch_size(e):
                print(
                    f"✗ [Iteration {iteration}] "
                    f"token_batch_size={token_batch_size} failed (OOM)"
                )
                clear_device_cache(garbage_collection=True)
                token_batch_size = int(token_batch_size * 0.7)

                if successful_batch_size is not None:
                    break
            else:
                raise

    if successful_batch_size is None:
        raise RuntimeError("Could not find a working token_batch_size")

    final_batch_size = successful_batch_size
    if round_to_pow2:
        final_batch_size = round_to_power_of_2(successful_batch_size)
        print(f"Rounded {successful_batch_size} → {final_batch_size} (power of 2)")

    print(f"\n{'='*60}")
    print(f"Optimal token_batch_size found: {final_batch_size}")
    print(f"{'='*60}\n")

    return final_batch_size


def find_optimal_token_batch_size(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    starting_batch_size: int = 4096,
) -> int:
    """
    Determine optimal token_batch_size for loaded models and data.

    This function assumes the model, tokenizer, and dataset are already loaded
    and ready to use. It will test different batch sizes to find the optimal
    token_batch_size that fits in available memory.

    Args:
        model: Already loaded and initialized model
        tokenizer: Already loaded tokenizer
        dataset: Small test dataset (already loaded)
        starting_batch_size: Starting batch size to test

    Returns:
        Optimal token_batch_size (power of 2)
    """
    print("\n" + "=" * 60)
    print("Finding optimal token_batch_size for loaded model...")
    print("=" * 60 + "\n")

    # Cap starting batch size to model's max sequence length
    max_seq_len = getattr(model.config, "max_position_embeddings", None)
    if max_seq_len is not None and starting_batch_size > max_seq_len:
        print(
            f"Capping starting_batch_size from {starting_batch_size} "
            f"to model's max sequence length {max_seq_len}"
        )
        starting_batch_size = max_seq_len

    processor = GradientProcessor(
        normalizers={},
        projection_dim=None,
        reshape_to_square=False,
        projection_type="rademacher",
    )

    def test_batch_size(token_batch_size: int) -> None:
        """Test function that tries a single forward/backward pass."""
        test_dataset = dataset.select(range(min(5, len(dataset))))

        test_dataset = test_dataset.map(
            tokenize,
            batched=True,
            fn_kwargs=dict(args=DataConfig(truncation=True), tokenizer=tokenizer),
        )
        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels", "length"]
        )

        index_cfg = IndexConfig(
            run_path="temp",
            model="test",
            token_batch_size=token_batch_size,
            loss_fn="ce",
            loss_reduction="mean",
        )

        test_collector = InMemoryCollector(
            model=model.base_model,  # type: ignore
            processor=processor,
            data=test_dataset,
            cfg=index_cfg,
        )

        batches = allocate_batches(test_dataset["length"], token_batch_size)  # type: ignore

        computer = CollectorComputer(
            model=model,
            data=test_dataset,
            collector=test_collector,
            batches=batches,
            cfg=index_cfg,
        )
        computer.run_with_collector_hooks(desc="batch size test")

    # Get max sequence length from model config
    max_seq_len = getattr(model.config, "max_position_embeddings", None)

    return find_optimal_token_batch_size_raw(
        test_fn=test_batch_size,
        starting_batch_size=starting_batch_size,
        round_to_pow2=True,
        max_batch_size=max_seq_len,
    )


def get_optimal_batch_size(
    cache_path: Path,
    model_hf_id: str,
    fsdp: bool,
    determine_fn: Callable[[], int],
) -> int:
    """
    Get optimal batch size from cache or determine it.

    Args:
        cache_path: Path to cache file
        model_hf_id: HuggingFace model ID
        fsdp: Whether FSDP is enabled
        starting_batch_size: Starting batch size for determination
        determine_fn: Function to determine batch size if not cached

    Returns:
        Optimal token_batch_size
    """
    # Try to load from cache
    cached_batch_size = load_batch_size_cache(cache_path, model_hf_id, fsdp)

    if cached_batch_size is not None:
        return cached_batch_size

    # Determine optimal batch size
    optimal_batch_size = determine_fn()

    # Save to cache
    save_batch_size_cache(cache_path, model_hf_id, optimal_batch_size, fsdp)

    return optimal_batch_size
