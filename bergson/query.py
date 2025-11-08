import os
import socket
from datetime import timedelta
from typing import cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset, IterableDataset
from peft import PeftConfig, PeftModel
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from torch.distributed.fsdp import fully_shard
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from .build import estimate_advantage
from .collection import collect_gradients
from .data import (
    IndexConfig,
    QueryConfig,
    allocate_batches,
    load_data_string,
    tokenize,
)
from .gradients import GradientProcessor
from .peft import detect_peft_modules
from .query_callback import get_scorer_callback
from .score_writer import MemmapScoreWriter
from .utils import assert_type, get_layer_list


def worker(
    rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    query_cfg: QueryConfig,
    ds: Dataset | IterableDataset,
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

    match index_cfg.precision:
        case "bf16":
            dtype = torch.bfloat16
        case "fp16":
            dtype = torch.float16
        case "fp32":
            dtype = torch.float32
        case "int4" | "int8":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        case "auto":
            dtype = "auto"
        case other:
            raise ValueError(f"Unsupported precision: {other}")

    device_map = {"": f"cuda:{rank}"} if not index_cfg.fsdp else "cpu"
    quantization_config = None
    if index_cfg.precision in ("int4", "int8"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=index_cfg.precision == "int4",
            load_in_8bit=index_cfg.precision == "int8",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Try to detect PEFT model
    try:
        peft_config = PeftConfig.from_pretrained(index_cfg.model)
    except ValueError:
        peft_config = None

    if peft_config is None:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            index_cfg.model,
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
            revision=index_cfg.revision,
        )
        target_modules = None

    else:
        # Load PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # type: ignore
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
            revision=index_cfg.revision,
        )

        model = PeftModel.from_pretrained(  # type: ignore
            base_model,
            index_cfg.model,
            device_map=device_map,
            autocast_adapter_dtype=False,
        )
        target_modules = detect_peft_modules(model)

        # Hack for type checking
        model = cast(PreTrainedModel, model)

    if rank == 0:
        print(f"Model loaded with dtype: {model.dtype}")

    embed = model.get_input_embeddings()
    model.requires_grad_(False)  # Freeze the model
    embed.requires_grad_(True)  # Make sure backward hooks are called though

    if index_cfg.fsdp:
        # Shard each individual transformer layer
        for layer in get_layer_list(model):
            fully_shard(layer)

        # Shard the entire model
        fully_shard(model)

    processor_dir = index_cfg.processor_path or index_cfg.run_path
    processor_cfg_path = os.path.join(processor_dir, "processor_config.json")

    if os.path.exists(processor_cfg_path):
        if rank == 0:
            print(f"Loading processor from '{processor_dir}'")

        processor = GradientProcessor.load(
            processor_dir,
            map_location=f"cuda:{rank}",
        )
    else:
        processor = GradientProcessor(
            {},
            projection_dim=index_cfg.projection_dim or None,
            reshape_to_square=index_cfg.reshape_to_square,
            projection_type=index_cfg.projection_type,
            include_bias=index_cfg.include_bias,
        )
        if rank == 0 and index_cfg.save_processor:
            processor.save(index_cfg.partial_run_path)

    if index_cfg.split_attention_modules:
        attention_cfgs = {
            module: index_cfg.attention for module in index_cfg.split_attention_modules
        }
    else:
        attention_cfgs = {}

    score_writer_dtype = dtype if dtype != "auto" else torch.float32
    if isinstance(ds, Dataset):

        scorer = get_scorer_callback(
            query_cfg,
            index_cfg.module_wise,
            torch.device(f"cuda:{rank}"),
            score_writer_dtype,
        )
        score_writer = MemmapScoreWriter(
            scorer,
            len(ds),
            query_cfg.scores_path,
            rank=rank,
            modules=query_cfg.modules,
            module_wise=index_cfg.module_wise,
            dtype=score_writer_dtype,
        )
        batches = allocate_batches(ds["length"][:], index_cfg.token_batch_size)
        collect_gradients(
            model,
            ds,
            processor,
            index_cfg.partial_run_path,
            batches=batches,
            kl_divergence=index_cfg.loss_fn == "kl",
            loss_reduction=index_cfg.loss_reduction,
            skip_preconditioners=index_cfg.skip_preconditioners,
            target_modules=target_modules,
            attention_cfgs=attention_cfgs,
            drop_columns=index_cfg.drop_columns,
            score_writer=score_writer,
            save_index=index_cfg.save_index,
            save_processor=index_cfg.save_processor,
            module_wise=index_cfg.module_wise,
        )
    else:
        # Convert each shard to a Dataset then collect its gradients
        buf, shard_id = [], 0

        def flush():
            nonlocal buf, shard_id
            if not buf:
                return
            ds_shard = assert_type(Dataset, Dataset.from_list(buf))
            batches = allocate_batches(
                ds_shard["length"][:], index_cfg.token_batch_size
            )
            scorer = get_scorer_callback(
                query_cfg,
                index_cfg.module_wise,
                torch.device(f"cuda:{rank}"),
                score_writer_dtype,
            )
            score_writer = MemmapScoreWriter(
                scorer,
                len(ds_shard),
                os.path.join(query_cfg.scores_path, f"shard-{shard_id:05d}"),
                rank=rank,
                modules=query_cfg.modules,
                module_wise=index_cfg.module_wise,
            )
            collect_gradients(
                model,
                ds_shard,
                processor,
                os.path.join(index_cfg.partial_run_path, f"shard-{shard_id:05d}"),
                batches=batches,
                kl_divergence=index_cfg.loss_fn == "kl",
                loss_reduction=index_cfg.loss_reduction,
                skip_preconditioners=index_cfg.skip_preconditioners,
                target_modules=target_modules,
                attention_cfgs=attention_cfgs,
                drop_columns=index_cfg.drop_columns,
                score_writer=score_writer,
                save_index=index_cfg.save_index,
                save_processor=index_cfg.save_processor,
            )
            buf.clear()
            shard_id += 1

        for ex in tqdm(ds, desc="Querying gradients on the fly", disable=rank != 0):
            buf.append(ex)
            if len(buf) == index_cfg.stream_shard_size:
                flush()
        flush()


def dist_worker(
    rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    query_cfg: QueryConfig,
    ds: Dataset,
):
    try:
        worker(rank, world_size, index_cfg, query_cfg, ds)
    finally:
        dist.destroy_process_group()


def query_gradient_dataset(query_cfg: QueryConfig, index_cfg: IndexConfig):
    # In many cases the token_batch_size may be smaller than the max length allowed by
    # the model. If cfg.data.truncation is True, we use the tokenizer to truncate
    tokenizer = AutoTokenizer.from_pretrained(
        index_cfg.model, revision=index_cfg.revision
    )
    tokenizer.model_max_length = min(
        tokenizer.model_max_length, index_cfg.token_batch_size
    )

    # Do all the data loading and preprocessing on the main process
    ds = load_data_string(
        index_cfg.data.dataset, index_cfg.data.split, streaming=index_cfg.streaming
    )

    remove_columns = ds.column_names if index_cfg.drop_columns else None
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=index_cfg.data, tokenizer=tokenizer),
        remove_columns=remove_columns,
    )
    if index_cfg.data.reward_column:
        assert isinstance(ds, Dataset), "Dataset required for advantage estimation"
        ds = ds.add_column(
            "advantage",
            estimate_advantage(ds, index_cfg.data),
            new_fingerprint="advantage",  # type: ignore
        )

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, index_cfg, query_cfg, ds)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = start_processes(
            "query",
            dist_worker,
            args={
                i: (i, world_size, index_cfg, query_cfg, ds) for i in range(world_size)
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

    try:
        os.rename(index_cfg.partial_run_path, index_cfg.run_path)
    except Exception:
        pass
