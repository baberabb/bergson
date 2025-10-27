import json
import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset, IterableDataset, Sequence, Value, load_from_disk
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
    load_gradient_dataset,
    tokenize,
)
from .gradients import GradientProcessor
from .peft import detect_peft_modules
from .query_writer import Query
from .utils import assert_type, get_layer_list


def get_query_data(query_cfg: QueryConfig):
    """
    Load and optionally precondition the query dataset. Preconditioners
    may be mixed as described in https://arxiv.org/html/2410.17413v1#S3.
    """
    # Collect the query gradients if they don't exist
    if not os.path.exists(query_cfg.query_path):
        raise FileNotFoundError(
            f"Query dataset not found at {query_cfg.query_path}. "
            "Please build a query dataset index first."
        )

    if query_cfg.query_path.endswith("full_accum_mean_mod_grads.hf"):
        print("Short circuiting code")
        query_ds = load_from_disk(query_cfg.query_path)
        if not query_cfg.modules:
            query_cfg.modules = list(query_ds.column_names)
            print(f"Modules: {query_cfg.modules}")
        query_ds = query_ds.with_format("torch", columns=query_cfg.modules)

        return query_ds

    # Load the query dataset
    with open(os.path.join(query_cfg.query_path, "info.json"), "r") as f:
        target_modules = json.load(f)["dtype"]["names"]

    query_ds = load_gradient_dataset(query_cfg.query_path, concatenate_gradients=False)
    query_ds = query_ds.with_format(
        "torch", columns=target_modules, dtype=torch.float32
    )

    # Ensure all gradient columns are materialized as float32 tensors with their native
    # sequence length.
    for col in target_modules:
        feature = query_ds.features[col]
        if isinstance(feature, Sequence):
            length = feature.length
        else:
            length = None
        query_ds = query_ds.cast_column(
            col,
            Sequence(Value("float32"), length=length),
        )
        print("Length", length)

    use_q = query_cfg.query_preconditioner_path is not None
    use_i = query_cfg.index_preconditioner_path is not None

    if use_q or use_i:
        q, i = {}, {}
        if use_q:
            assert query_cfg.query_preconditioner_path is not None
            q = GradientProcessor.load(
                query_cfg.query_preconditioner_path,
                map_location="cuda",
            ).preconditioners
        if use_i:
            assert query_cfg.index_preconditioner_path is not None
            i = GradientProcessor.load(
                query_cfg.index_preconditioner_path, map_location="cuda"
            ).preconditioners

        mixed_preconditioner = (
            {
                k: q[k] * query_cfg.mixing_coefficient
                + i[k] * (1 - query_cfg.mixing_coefficient)
                for k in q
            }
            if (q and i)
            else (q or i)
        )
        mixed_preconditioner = {
            k: v.cuda().to(torch.float64) for k, v in mixed_preconditioner.items()
        }

        def precondition(batch):
            print("batch")
            for name in target_modules:
                result = (
                    batch[name].cuda().to(torch.float64) @ mixed_preconditioner[name]
                )
                batch[name] = result.cpu()

            return batch

        query_ds = query_ds.map(
            precondition, batched=True, batch_size=query_cfg.batch_size
        )

        for name in target_modules:
            assert query_ds[0][name].sum() != 0

    return query_ds


def get_individual_query(
    query_ds: Dataset, query_cfg: QueryConfig, device: torch.device, dtype: torch.dtype
):
    """
    Compute the individual query and return a callback function that scores gradients
    according to their inner products or cosine similarities with the individual
    queries.
    """
    queries = torch.cat([query_ds[:][name] for name in query_cfg.modules], dim=1).to(
        device=device, dtype=dtype
    )

    if query_cfg.unit_normalize:
        queries /= queries.norm(dim=1, keepdim=True)

    # Assert on device
    assert queries.device == device

    def callback(mod_grads: dict[str, torch.Tensor]):
        grads = torch.cat([mod_grads[name] for name in query_cfg.modules], dim=1)
        if query_cfg.unit_normalize:
            grads /= grads.norm(dim=1, keepdim=True)

        # Return a score for every query
        return grads @ queries.T

    return callback


def get_module_wise_mean_query(
    query_ds: Dataset,
    query_cfg: QueryConfig,
    device: torch.device,
    dtype: torch.dtype,
    precomputed_mean: bool = True,
):
    """
    Compute the mean query and return a callback function that scores gradients
    according to their inner products or cosine similarities with the mean query.
    """
    # Accumulate on CPU to avoid holding full-resolution gradients on GPU.
    if not precomputed_mean:
        acc_device = torch.device("cpu")
        acc = {
            module: torch.zeros_like(
                query_ds[0][module], device=acc_device, dtype=torch.float32
            )
            for module in query_cfg.modules
        }

        def sum_(*cols):
            for module, x in zip(query_cfg.modules, cols):
                x = x.to(device=acc_device, dtype=torch.float32)
                if query_cfg.unit_normalize:
                    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
                acc[module].add_(x.sum(0))

        query_ds.map(
            sum_,
            input_columns=query_cfg.modules,
            batched=True,
            batch_size=query_cfg.batch_size,
        )

        callback_query = {
            module: (acc[module] / len(query_ds)).to(
                device=device, dtype=dtype, non_blocking=True
            )
            for module in query_cfg.modules
        }
    else:
        if query_cfg.unit_normalize:
            callback_query = {
                module: query_ds[0][module].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                for module in query_cfg.modules
            }
            norm = (
                torch.cat(
                    [
                        (query_ds[0][module].to(torch.float32) ** 2).sum(dim=1)
                        for module in query_cfg.modules
                    ],
                    dim=0,
                )
                .sum()
                .sqrt()
            )
            print(norm.shape)
            callback_query = {
                module: (callback_query[module].to(torch.float32) / norm).to(
                    dtype=dtype
                )
                for module in query_cfg.modules
            }
        else:
            callback_query = {
                module: query_ds[0][module].to(
                    device=device, dtype=dtype, non_blocking=True
                )
                for module in query_cfg.modules
            }

    @torch.inference_mode()
    def callback(mod_grads: dict[str, torch.Tensor], name: str):
        module_scores = torch.inner(mod_grads[name], callback_query[name]).to(
            "cpu", non_blocking=True
        )
        mod_grads[name].pow_(2)
        sum_of_squares = mod_grads[name].sum(dim=1).to("cpu", non_blocking=True)

        return module_scores, sum_of_squares

    return callback


def get_mean_query(
    query_ds: Dataset, query_cfg: QueryConfig, device: torch.device, dtype: torch.dtype
):
    """
    Compute the mean query and return a callback function that scores gradients
    according to their inner products or cosine similarities with the mean query.
    """
    # Accumulate on CPU to avoid holding full-resolution gradients on GPU.
    acc_device = torch.device("cpu")
    acc = {
        module: torch.zeros_like(
            query_ds[0][module], device=acc_device, dtype=torch.float32
        )
        for module in query_cfg.modules
    }

    def sum_(*cols):
        for module, x in zip(query_cfg.modules, cols):
            x = x.to(device=acc_device, dtype=torch.float32)
            if query_cfg.unit_normalize:
                x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
            acc[module].add_(x.sum(0))

    query_ds.map(
        sum_,
        input_columns=query_cfg.modules,
        batched=True,
        batch_size=query_cfg.batch_size,
    )

    callback_query = torch.cat(
        [
            (acc[module] / len(query_ds)).to(
                device=device, dtype=dtype, non_blocking=True
            )
            for module in query_cfg.modules
        ],
        dim=0,
    )

    @torch.inference_mode()
    def callback(mod_grads: dict[str, torch.Tensor]):
        grads = torch.cat([mod_grads[name] for name in query_cfg.modules], dim=1)
        if query_cfg.unit_normalize:
            grads /= grads.norm(dim=1, keepdim=True)
        return grads @ callback_query

    return callback


def get_nearest_query(
    query_ds: Dataset, query_cfg: QueryConfig, device: torch.device, dtype: torch.dtype
):
    """
    Return a callback function that scores gradients according to their cosine
    similarities or inner products with the most similar gradient in the query
    dataset.
    """

    queries = torch.cat([query_ds[:][name] for name in query_cfg.modules], dim=1).to(
        device=device, dtype=dtype
    )

    if query_cfg.unit_normalize:
        queries /= queries.norm(dim=1, keepdim=True)

    def callback(mod_grads: dict[str, torch.Tensor]):
        grads = torch.cat([mod_grads[name] for name in query_cfg.modules], dim=1)
        if query_cfg.unit_normalize:
            grads /= grads.norm(dim=1, keepdim=True)

        # Calculate scores as the max of the inner products with the queries
        all_scores = grads @ queries.T
        return all_scores.max(dim=-1).values

    return callback


def filter_complete_indices(
    index_cfg: IndexConfig,
    query_cfg: QueryConfig,
    batches: list[list[int]],
    rank: int,
    rows_per_file: int = 10_000_000,
):
    """
    Filter out indices that are present in the scores.csv file.
    """

    # find and concatenate all the scores.csv files into a single dataframe
    import pandas as pd

    dfs_dir = Path(query_cfg.scores_path) / f"rank_{rank}"
    available_dfs = [
        pd.read_csv(dfs_dir / file)
        for file in sorted(os.listdir(dfs_dir))
        if file.endswith(".csv")
    ]
    scores_df = pd.concat(available_dfs)

    if scores_df.empty:
        return batches

    # Archive unfiltered data
    raw_scores_csv_path = Path(query_cfg.scores_path) / f"rank_{rank}_raw"
    for i in range(0, len(scores_df), rows_per_file):
        scores_df.iloc[i : i + rows_per_file].to_csv(
            str(raw_scores_csv_path / f"scores_{i:02d}.csv")
        )

    target_rows_per_index = len(query_cfg.modules) if index_cfg.module_wise else 1
    original_scores_df_len = len(scores_df)

    # Remove rows for indices with fewer than target_rows_per_index

    # Group by indices and filter out indices with fewer than
    # target_rows_per_index rows
    scores_df = scores_df.groupby("index").filter(
        lambda x: len(x) >= target_rows_per_index
    )
    scores_df = assert_type(pd.DataFrame, scores_df)
    # Convert back to a csv and save
    for i in range(0, len(scores_df), rows_per_file):
        scores_df.iloc[i : i + rows_per_file].to_csv(
            str(Path(query_cfg.scores_path) / f"rank_{rank}" / f"scores_{i:02d}.csv")
        )

    print(
        f"Removed {original_scores_df_len - len(scores_df)} rows "
        f"from rank_{rank} scores.csv file"
    )

    scores_indices = scores_df["index"].unique().tolist()
    print(f"Found {len(scores_indices)} indices in rank_{rank} scores.csv file")

    batches = [
        [idx for idx in batch if idx not in scores_indices]
        for batch in batches
        if len(batch) > 0
    ]
    print(f"Filtered {len(scores_indices)} indices from batches")

    return batches


def worker(
    rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    query_cfg: QueryConfig,
    ds: Dataset | IterableDataset,
    query_ds: Dataset,
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
            timeout=timedelta(minutes=20),
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
        )
        if index_cfg.save_processor:
            if rank == 0:
                processor.save(index_cfg.partial_run_path)

    if index_cfg.split_attention_modules:
        attention_cfgs = {
            module: index_cfg.attention for module in index_cfg.split_attention_modules
        }
    else:
        attention_cfgs = {}

    if not query_cfg.modules:
        with open(os.path.join(query_cfg.query_path, "info.json"), "r") as f:
            query_cfg.modules = json.load(f)["dtype"]["names"]

    query_ds = query_ds.with_format("torch", columns=query_cfg.modules)

    query_dtype = dtype if dtype != "auto" else torch.float16

    if query_cfg.score == "mean":
        if index_cfg.module_wise:
            base_query_callback = get_module_wise_mean_query(
                query_ds, query_cfg, model.device, query_dtype
            )
        else:
            base_query_callback = get_mean_query(
                query_ds, query_cfg, model.device, query_dtype
            )
        num_scores = 1
    elif query_cfg.score == "nearest":
        base_query_callback = get_nearest_query(
            query_ds, query_cfg, model.device, query_dtype
        )
        num_scores = 1
    elif query_cfg.score == "individual":
        base_query_callback = get_individual_query(
            query_ds, query_cfg, model.device, query_dtype
        )
        num_scores = len(query_ds)
    else:
        raise ValueError(f"Invalid query scoring method: {query_cfg.score}")

    scores_dtype = torch.float32 if model.dtype == torch.float32 else torch.float16

    if isinstance(ds, Dataset):
        batches = allocate_batches(ds["length"][:], index_cfg.token_batch_size)
        batches = filter_complete_indices(index_cfg, query_cfg, batches, rank)

        query = Query(
            base_query_callback,
            len(ds),
            num_scores,
            str(Path(query_cfg.scores_path)),
            dtype=scores_dtype,
            device=model.device,
            rank=rank,
            module_wise=index_cfg.module_wise,
        )
        collect_gradients(
            model,
            ds,
            processor,
            index_cfg.partial_run_path,
            batches=batches,
            token_batch_size=index_cfg.token_batch_size,
            kl_divergence=index_cfg.loss_fn == "kl",
            loss_reduction=index_cfg.loss_reduction,
            skip_preconditioners=index_cfg.skip_preconditioners,
            target_modules=target_modules,
            attention_cfgs=attention_cfgs,
            drop_columns=index_cfg.drop_columns,
            query=query,
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
            query = Query(
                base_query_callback,
                len(ds_shard),
                num_scores,
                str(Path(query_cfg.scores_path) / f"shard-{shard_id:05d}"),
                dtype=scores_dtype,
                device=model.device,
                rank=rank,
                module_wise=index_cfg.module_wise,
            )
            collect_gradients(
                model,
                ds_shard,
                processor,
                os.path.join(index_cfg.partial_run_path, f"shard-{shard_id:05d}"),
                token_batch_size=index_cfg.token_batch_size,
                batches=batches,
                kl_divergence=index_cfg.loss_fn == "kl",
                loss_reduction=index_cfg.loss_reduction,
                skip_preconditioners=index_cfg.skip_preconditioners,
                target_modules=target_modules,
                attention_cfgs=attention_cfgs,
                drop_columns=index_cfg.drop_columns,
                query=query,
                save_index=index_cfg.save_index,
                save_processor=index_cfg.save_processor,
                module_wise=index_cfg.module_wise,
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
    query_ds: Dataset,
):
    try:
        worker(rank, world_size, index_cfg, query_cfg, ds, query_ds)
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

    os.makedirs(index_cfg.partial_run_path, exist_ok=True)

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

    query_ds = get_query_data(query_cfg)

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, index_cfg, query_cfg, ds, query_ds)
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
                i: (i, world_size, index_cfg, query_cfg, ds, query_ds)
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

    try:
        os.rename(index_cfg.partial_run_path, index_cfg.run_path)
    except Exception as e:
        print(f"Error renaming index path: {e}")
