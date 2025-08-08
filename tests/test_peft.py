import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.utils import get_layer_list


def setup_model_and_peft(fsdp, lora) -> tuple[AutoModelForCausalLM, set | None]:
    """Handle model loading, quantization, FSDP, and PEFT detection"""
    world_size = torch.cuda.device_count() if fsdp else 1

    rank = int(os.environ.get("RANK", "0"))
    print(rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            init_method="tcp://localhost:29500",  # Changed init method
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    torch.cuda.set_device(rank)

    model_str = "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-medical/checkpoint-793"
    # model_str = "Qwen/Qwen1.5-1.8B"
    # model_str = "Qwen/Qwen2.5-14B"

    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        device_map={"": f"cuda:{rank}" if not fsdp else "cpu"},
        torch_dtype=torch.bfloat16,
    )

    # Check for PEFT adapters

    if lora:
        adapters = model.active_adapters()

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

    if fsdp:
        # Shard each individual transformer layer
        for layer in get_layer_list(model):
            fully_shard(layer)

        # Shard the entire model
        fully_shard(model)

    # test_input = torch.randint(3, [1, 256, 256], device=f"cuda:{rank}")
    example_text = "This is a test input"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    test_input = tokenizer(
        example_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
    ).to(f"cuda:{rank}")

    print(test_input["input_ids"].device)
    logits = model(**test_input).logits

    logits.sum().backward()


if __name__ == "__main__":
    setup_model_and_peft(fsdp=True, lora=False)
