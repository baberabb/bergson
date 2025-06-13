import os

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from simple_parsing import parse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data import IndexConfig, MemmapDataset, compute_batches, tokenize
from .gradients import GradientProcessor
from .processing import build_index, fit_normalizers
from .utils import assert_type


def run():
    args = parse(IndexConfig)

    # Initialize distributed training
    if os.environ.get("LOCAL_RANK") is not None:
        dist.init_process_group("nccl")

    # Set the random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    torch.cuda.set_device(rank)

    dtype = None
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None
        ),
        torch_dtype=dtype,
    )

    # Check for PEFT adapters
    try:
        adapters = model.active_adapters()
    except ValueError:
        target_modules = None
    else:
        if rank == 0:
            print("PEFT model detected.")

        target_modules = set()

        for adapter_name in adapters:
            state = model.get_adapter_state_dict(adapter_name)

            for name in state:
                prefix = name.removesuffix(".weight")
                name = prefix + "." + adapter_name

                try:
                    model.get_submodule(name)
                except AttributeError:
                    print(f"Adapter parameter '{name}' not found in the model.")

                target_modules.add(name.removeprefix("model."))

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    if args.dataset.endswith(".bin"):
        # TODO: Make this configurable, right now this is just a hack to support
        # the Pythia preshuffled Pile dataset.
        MEMMAP_CTX_LEN = 2049

        # If the dataset is a memmap file, use MemmapDataset
        ds = MemmapDataset(args.dataset, MEMMAP_CTX_LEN)
        ds = ds.shard(world_size, rank)

        # Uniform batches
        batch_size = args.token_batch_size // MEMMAP_CTX_LEN
        batches = [
            slice(start, start + batch_size) for start in range(0, len(ds), batch_size)
        ]
    else:
        try:
            ds = assert_type(Dataset, load_dataset(args.dataset, split="train"))
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                ds = Dataset.load_from_disk(args.dataset, keep_in_memory=False)
            else:
                raise e

        metadata = {"length"}
        if args.drop_columns:
            metadata |= set(ds.column_names)

        assert (
            "row_number" not in ds.column_names
        ), "The dataset already contains a column named 'row_number'. "

        ds = ds.map(lambda _, idx: dict(row_number=idx), with_indices=True)
        ds = ds.shuffle(seed=42).shard(world_size, rank)

        # Shuffle before sharding to make sure each rank gets a different subset
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        ds = ds.map(
            tokenize,
            batched=True,
            fn_kwargs=dict(args=args, tokenizer=tokenizer),
        )
        ds = ds.sort("length", reverse=True)
        batches = compute_batches(ds["length"], args.token_batch_size)

        ds = ds.remove_columns(list(metadata))

    if os.path.exists(args.processor_path):
        if rank == 0:
            print(f"Loading processor from '{args.processor_path}'")

        processor = GradientProcessor.load(
            args.processor_path,
            map_location=f"cuda:{rank}",
        )
    else:
        if args.normalizer != "none":
            normalizers = fit_normalizers(
                model,
                ds,
                batches=batches,
                kind=args.normalizer,
                max_documents=args.stats_sample_size or None,
                target_modules=target_modules,
            )
        else:
            normalizers = {}

        processor = GradientProcessor(
            normalizers,
            fisher_fourth_root=args.fisher_fourth_root,
            projection_dim=args.projection_dim or None,
        )
        # processor.estimate_preconditioners(
        #     model,
        #     ds,
        #     batches=batches,
        #     max_documents=args.stats_sample_size or None,
        #     target_modules=target_modules,
        # )
        processor.save(args.run_path)

    # Build the index
    build_index(
        model,
        ds,
        processor,
        args.run_path,
        batches=batches,
        target_modules=target_modules,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
