import math
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from simple_parsing import parse
from torch import Tensor
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from trl import SFTConfig, SFTTrainer, setup_chat_format

from bergson import (
    GradientCollector,
    GradientProcessor,
)
from bergson.data import (
    DataConfig,
    IndexConfig,
    load_data_string,
    tokenize,
)
from bergson.huggingface import (
    GradientCollectorCallback,
    prepare_for_gradient_collection,
)


def configure_gradient_collection(
    cfg: IndexConfig,
    model: PreTrainedModel,
    rank: int,
    train: Dataset,
):
    # Set up closure for collecting gradients
    lo = torch.finfo(torch.float16).min
    hi = torch.finfo(torch.float16).max

    mod_grads = {}
    preconditioners = {}
    skip_preconditioners = False

    def closure(name: str, g: Tensor):
        """Send model gradients for a module to self.mod_grads and
        update preconditioners."""
        g = g.flatten(1).clamp_(lo, hi)

        # Asynchronously move the gradient to CPU and convert to fp16
        mod_grads[name] = g.to(device="cpu", dtype=torch.float16, non_blocking=True)

        # Compute the outer product of the flattened gradient
        if not skip_preconditioners:
            g = g.float()
            preconditioner = preconditioners.get(name, None)
            if preconditioner is None:
                preconditioners[name] = g.mT @ g
            else:
                preconditioners[name].addmm_(g.mT, g)

    # Set up normalizers, preconditioners, and target modules
    try:
        adapters = model.active_adapters()
    except ValueError:
        target_modules = None
    else:
        cfg.normalizer = "adam"
        cfg.reshape_to_square = True

        if rank == 0:
            print("PEFT model detected. Using Adam and reshape_to_square = True")

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

    if os.path.exists(cfg.processor_path):
        if rank == 0:
            print(f"Loading processor from '{cfg.processor_path}'")

        processor = GradientProcessor.load(
            cfg.processor_path,
            map_location=f"cuda:{rank}",
        )
    else:
        processor = GradientProcessor(
            {},
            fisher_fourth_root=cfg.fisher_fourth_root,
            projection_dim=cfg.projection_dim or None,
            reshape_to_square=cfg.reshape_to_square,
        )
        if rank == 0:
            processor.save(cfg.run_path)

    return closure, processor, mod_grads, preconditioners, target_modules


def worker(
    rank: int,
    world_size: int,
    cfg: IndexConfig,
    train: Dataset,
    eval: Dataset,
    run_name,
):
    torch.cuda.set_device(rank)

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype="bfloat16",
        revision=cfg.revision,
        device_map=f"cuda:{rank}",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, max_length=8192)
    model, tokenizer = setup_chat_format(model, tokenizer)

    closure, processor, mod_grads, preconditioners, target_modules = (
        configure_gradient_collection(cfg, model, rank, train)
    )

    gradient_collector = GradientCollector(
        model=model.base_model,
        closure=closure,
        processor=processor,
        target_modules=target_modules,
    )

    # Create the callback
    grad_sizes = {name: math.prod(s) for name, s in gradient_collector.shapes().items()}

    gradient_collector_callback = GradientCollectorCallback(
        collector=gradient_collector,
        processor=processor,
        mod_grads=mod_grads,
        preconditioners=preconditioners,
        grad_sizes=grad_sizes,
        path=f"examples/runs/gradients/{run_name}",
        mean_gradients=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train,  # type: ignore
        eval_dataset=eval,
        callbacks=[gradient_collector_callback],
        args=SFTConfig(
            max_length=8192,
            output_dir=f"examples/runs/{run_name}",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            num_train_epochs=2,
            logging_steps=10,
            eval_steps=20,
            save_total_limit=3,
            group_by_length=True,
            completion_only_loss=True,
            ddp_find_unused_parameters=False,
            seed=0,
        ),
    )
    trainer = prepare_for_gradient_collection(trainer)

    with gradient_collector:
        trainer.train()


def dist_worker(
    rank: int,
    world_size: int,
    cfg: IndexConfig,
    train: Dataset,
    eval: Dataset,
    run_name: str,
):
    try:
        worker(rank, world_size, cfg, train, eval, run_name)
    finally:
        dist.destroy_process_group()


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(args: IndexConfig):
    seed = 0
    set_seeds(seed)

    run_name = (
        f"{args.model.split('/')[-1]}-{args.data.dataset.split('/')[-1]}" f"-s={seed}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="bfloat16",
        revision=args.revision,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=8192)
    model, tokenizer = setup_chat_format(model, tokenizer)

    # Create DataConfig for tokenization
    data_config = DataConfig(
        prompt_column=args.data.prompt_column,
        completion_column=args.data.completion_column,
        conversation_column=args.data.conversation_column,
    )
    dataset = load_data_string(
        args.data.dataset, args.data.split, streaming=args.streaming
    )
    dataset = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
    )

    train, eval = dataset.train_test_split(
        test_size=0.05,
        seed=seed,
    ).values()

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, args, train, eval, run_name)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = start_processes(
            "train",
            dist_worker,
            args={
                i: (i, world_size, args, train, eval, run_name)
                for i in range(world_size)
            },
            envs={
                i: {
                    "LOCAL_RANK": str(i),
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": str(port),
                }
                for i in range(world_size)
            },
            logs_specs=DefaultLogsSpecs(),
        )
        ctx.wait()


if __name__ == "__main__":
    args = parse(IndexConfig)

    main(args)
