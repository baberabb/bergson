"""
CLI tool for automatically determining optimal token_batch_size.

This runs as a standalone command before benchmarks to avoid issues with
multi-GPU/distributed training.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import field

from ..utils.auto_batch_size import (
    determine_batch_size_cli,
    determine_batch_size_disk,
    get_optimal_batch_size,
    load_batch_size_cache,
)


@dataclass
class AutoBatchSizeConfig:
    """Configuration for auto batch size determination."""

    model: str = field(positional=True)
    """HuggingFace model ID (e.g., EleutherAI/pythia-70m)."""

    output: str = field(positional=True)
    """Path to save batch size cache (e.g., runs/my_run/batch_size_cache.json)."""

    dataset: str = "Skylion007/openwebtext"
    """Dataset to use for testing."""

    split: str = "train"
    """Dataset split to use for testing."""

    max_length: int = 1024
    """Maximum sequence length."""

    starting_batch_size: int = 16384
    """Starting token_batch_size to try (will be optimized down to power of 2)."""

    fsdp: bool = False
    """Enable FSDP for model loading."""

    method: str = "disk"
    """Method to use: 'disk' (loads model in-memory), 'cli' (uses bergson build subprocess)."""

    overwrite: bool = False
    """Overwrite the cache."""


@dataclass
class AutoBatchSize:
    """Auto-determine optimal token_batch_size for hardware and model."""

    config: AutoBatchSizeConfig

    def execute(self):
        """Determine optimal token_batch_size."""
        output_path = Path(self.config.output).resolve()

        # Check cache unless force is specified
        if not self.config.overwrite and output_path.exists():
            cached = load_batch_size_cache(
                output_path, self.config.model, self.config.fsdp
            )
            if cached is not None:
                print(f"\n✓ Using cached token_batch_size={cached}")
                print("  (use --overwrite to re-determine)")
                return

        # Determine optimal batch size
        if self.config.method == "disk":

            def determine_fn():
                return determine_batch_size_disk(
                    model_hf_id=self.config.model,
                    dataset_name=self.config.dataset,
                    dataset_split=self.config.split,
                    max_length=self.config.max_length,
                    starting_batch_size=self.config.starting_batch_size,
                    use_fsdp=self.config.fsdp,
                )

        elif self.config.method == "cli":
            # Create temporary test path
            test_path = output_path.parent / "auto_batch_size_test"

            def determine_fn():
                result = determine_batch_size_cli(
                    model_hf_id=self.config.model,
                    dataset=self.config.dataset,
                    test_path=test_path,
                    starting_batch_size=self.config.starting_batch_size,
                    use_fsdp=self.config.fsdp,
                )

                # Clean up test directory
                if test_path.exists():
                    shutil.rmtree(test_path)

                return result

        else:
            raise ValueError(
                f"Unknown method: {self.config.method}. Use 'disk' or 'cli'."
            )

        optimal_batch_size = get_optimal_batch_size(
            cache_path=output_path,
            model_hf_id=self.config.model,
            fsdp=self.config.fsdp,
            starting_batch_size=self.config.starting_batch_size,
            determine_fn=determine_fn,
        )

        print(f"\n{'='*60}")
        print(f"✓ Optimal token_batch_size: {optimal_batch_size}")
        print(f"✓ Saved to: {output_path}")
        print(f"{'='*60}\n")
        print("Use this in your benchmark with:")
        print(f"  --token_batch_size {optimal_batch_size}")
        print(
            "\nOr enable auto_batch_size in benchmarks to load from cache automatically."
        )
