from typing import Literal
import os
import heapq
from dataclasses import dataclass
from contextlib import nullcontext, redirect_stdout

from simple_parsing import parse
from tqdm import tqdm
from torch import Tensor
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, setup_chat_format

from bergson.data import load_gradient_dataset, tokenize, DataConfig
from bergson.utils import assert_type
import numpy as np


@dataclass
class FilterConfig():
    """Config for building the index and running the model/dataset pipeline."""
    filter: Literal["classification", "attribution", "loss", "random"] = "attribution"
    """Filter to apply to the training set before finetuning."""

    model: str = "HuggingFaceTB/SmolLM2-1.7B"
    """Name of the model to load."""

    dataset: str = "argilla/magpie-ultra-v0.1"
    """Dataset identifier to finetune on."""

    dataset_index: str = ""
    """Bergson index to use for attribution and loss filtering."""

    query_index: str = ""
    """
    The mean of this index's gradients will be used to query the dataset index for 
    attribution filtering. When unspecified the query is the mean of the value gradients.
    """

    name: str | None = None
    """Name of the run, used to save the model and tokenizer."""

    num_examples: int = 30_000
    """Number of items to select from the training set."""

    lowest: bool = False
    """Select the lowest scores."""

    prompt_column: str = "text"
    """Column in the dataset that contains the prompts."""

    completion_column: str = ""
    """Optional column in the dataset that contains the completions."""

    conversation_column: str = ""
    """Optional column in the dataset that contains the conversation."""

    seed: int = 42
    """Seed for reproducibility."""


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def select_topk(ds: Dataset, n: int, key: str, lowest: bool = False):
    heap = []

    for idx, s in enumerate(ds[key]):
        key = -s if lowest else s

        if len(heap) < n:
            heapq.heappush(heap, (key, idx))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, idx))
    return ds.select([i for _, i in heap])


def normalize_batch(batch):
    return {
        "gradients": F.normalize(batch["gradients"].cuda(), dim=1).cpu(),
    }


def add_index(
    dataset: Dataset,
    index: str | None = None,
) -> Dataset:
    assert index is not None, "Index must be provided for attribution or loss filtering"

    gradient_ds = (
        load_gradient_dataset(index)
        .with_format("torch")
    )
    def generator():
        for row, grad_row in zip(dataset, gradient_ds):
            yield {**dict(row), **dict(grad_row)}

    return assert_type(Dataset, Dataset.from_generator(generator))


def main(
    args: FilterConfig,
):
    set_seeds(args.seed)

    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)

    if args.name is None:
        run_name = (
            f"{args.model.split('/')[-1]}-{args.dataset.split('/')[-1]}-{args.filter}"
            f"{'-lowest' if args.lowest else ''}"
            f"-s={args.seed}"
        )
    else:
        run_name = args.name

    with nullcontext() if rank == 0 else redirect_stdout(None):
        if args.query_index and args.filter == "attribution":
            print("loading index...")
            query_index = (
                load_gradient_dataset(args.query_index)
                    .with_format("torch")
                    .map(normalize_batch, batched=True, batch_size=512)      
            )
            print("query index loaded")
            query = assert_type(Tensor, query_index["gradients"]).mean(0).cuda()
            del query_index
        else:
            query = None
        
        dataset = assert_type(Dataset, load_dataset(args.dataset, split="train"))

        if args.filter == "attribution" or args.filter == "loss":
            dataset = add_index(dataset, args.dataset_index)

        dataset.shuffle(args.seed).with_format("torch")

        train, eval = dataset.train_test_split(
            test_size=0.05,
            seed=args.seed,
            load_from_cache_file=True,
            train_indices_cache_file_name=f"cache/{run_name}/train.arrow",
            test_indices_cache_file_name=f"cache/{run_name}/test.arrow",
        ).values()

        train.set_format("torch")
        eval.set_format("torch")

        print("Filtering...")
        if args.num_examples == 0:
            pass
        elif args.filter == "attribution":
            if query == None:
                # Calculate mean gradient in batches to avoid memory issues
                print("Calculating mean gradient...")
                mean_gradient = None
                batch_size = 100  # Process in smaller batches
                
                for i in tqdm(range(0, len(train), batch_size), desc="Computing mean gradient"):
                    batch = train.select(range(i, min(i + batch_size, len(train))))
                    batch_gradients = torch.stack([g for g in batch["gradients"]])
                    
                    if mean_gradient is None:
                        mean_gradient = batch_gradients.mean(0)
                    else:
                        mean_gradient = (mean_gradient * i + batch_gradients.sum(0)) / (i + len(batch))
                    
                    # Clear memory
                    del batch_gradients
                
                query = mean_gradient.cuda()
            
            print("Computing importance scores...")
            # Calculate importance scores in batches
            importance_scores = []
            batch_size = 512
            
            for i in tqdm(range(0, len(train), batch_size), desc="Computing importance scores"):
                batch = train.select(range(i, min(i + batch_size, len(train))))
                batch_gradients = torch.stack([g for g in batch["gradients"]]).cuda()
                
                # Normalize gradients
                batch_gradients = F.normalize(batch_gradients, dim=1)
                
                # Compute inner products
                batch_scores = (batch_gradients @ query).cpu().numpy()
                importance_scores.extend(batch_scores)
                
                # Clear GPU memory
                del batch_gradients
                torch.cuda.empty_cache()
            
            # Convert to numpy array for efficient sorting
            importance_scores = np.array(importance_scores)
            
            # Get top-k indices
            if args.lowest:
                selected_indices = np.argsort(importance_scores)[:args.num_examples]
            else:
                selected_indices = np.argsort(importance_scores)[-args.num_examples:]
            
            # Select the top-k examples
            train = train.select(selected_indices)
            
            # Clear memory
            del importance_scores, query
            torch.cuda.empty_cache()
        elif args.filter == "classification":
            ranks = {"excellent": 4, "good": 3, "average": 2, "poor": 1, "very poor": 0}

            def add_rank(ex):
                q = ex.get("quality")
                return {"_q": ranks.get(q, -1)}

            train = (
                train.map(add_rank)
                .filter(lambda x: x["_q"] >= 0)
                .sort("_q", reverse=not args.lowest)
                .select(range(min(args.num_examples, len(train))))
                .remove_columns("_q")
            )
        elif args.filter == "loss":
            train = train.map(
                lambda x: {"loss": x["loss"].mean().item()},
                remove_columns=["gradients"],
            )
            train = select_topk(train, args.num_examples, "loss", lowest=args.lowest)
        elif args.filter == "random":
            train = train.select(range(min(args.num_examples, len(train))))
        else:
            raise ValueError(f"Invalid filter: {args.filter}")

        # Clear memory before loading model
        torch.cuda.empty_cache()
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="bfloat16",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=8192)
        model, tokenizer = setup_chat_format(model, tokenizer)

        # Create DataConfig for tokenization
        data_config = DataConfig(
            prompt_column=args.prompt_column,
            completion_column=args.completion_column,
            conversation_column=args.conversation_column,
        )

        train = train.map(
            tokenize,
            batched=True,
            fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
        )
        eval = eval.map(
            tokenize,
            batched=True,
            fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
        )

        # https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/sft/config.yaml
        trainer = SFTTrainer(
            model=model,
            train_dataset=train,
            eval_dataset=eval,
            args=SFTConfig(
                max_length=8192,
                max_seq_length=8192,
                output_dir=f"examples/runs/{run_name}",
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=32,
                gradient_checkpointing=True,
                learning_rate=3e-4,
                num_train_epochs=1,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                bf16=True,
                logging_steps=10,
                eval_steps=20,
                save_steps=100,
                save_total_limit=3,
                group_by_length=True,
                completion_only_loss=True,
                ddp_find_unused_parameters=False,
                seed=args.seed,
            ),
        )

        trainer.train()

        if rank == 0:
            trainer.save_model(f"examples/runs/{run_name}")
            tokenizer.save_pretrained(f"examples/runs/{run_name}")


if __name__ == "__main__":
    args = parse(FilterConfig)

    main(
        args
    )
