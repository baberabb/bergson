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

    name: str | None = None
    """Name of the run, used to save the model and tokenizer."""

    num_examples: int = 30_000
    """Number of items to select from the training set."""

    prompt_column: str = "text"
    """Column in the dataset that contains the prompts."""

    completion_column: str = ""
    """Optional column in the dataset that contains the completions."""

    conversation_column: str = ""
    """Optional column in the dataset that contains the conversation."""

    batch_size: int = 512
    """Batch size for processing the dataset."""

    seed: int = 42
    """Seed for reproducibility."""

    query_index: str = ""
    """
    Use the mean of this index's gradients as the query for attribution 
    filtering. If unspecified the query is calculated over the dataset 
    index.
    """

    lowest: bool = False
    """Select the lowest scores."""

    sample: bool = False
    """Filter by sampling from the dataset without replacement with 
    probability proportional to the filtering criteria."""

    temperature: float = 0.1
    """Temperature for sampling, used to control the distribution of
    the sampling probabilities. Lower values make the distribution more
    uniform, while higher values make it more peaked."""

    num_epochs: int = 1
    """Number of epochs to train for."""

    hf_token: str | None = None
    """Hugging Face token to use for the dataset."""


def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def get_mean_normalized_gradients(
    dataset: Dataset,
    batch_size: int = 512,
) -> Tensor:
    """Compute the mean of the gradients in the dataset."""
    gradients_sum = torch.zeros(dataset['gradients'][1].shape, device="cuda")
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Computing mean gradient"):
        gradients_batch = dataset[i : min(i + batch_size, len(dataset))]['gradients'].cuda()
        gradients_sum += gradients_batch.sum(0)
        
        del gradients_batch
    
    return gradients_sum / len(dataset)


def attribution_filter(
    args: FilterConfig,
    train: Dataset,
) -> Dataset:
    query_dataset = (
        load_gradient_dataset(args.query_index).with_format("torch")
        if args.query_index
        else train
    )
    # Compute the mean of the normalized gradients in the query dataset
    query = get_mean_normalized_gradients(query_dataset, args.batch_size)
            
    importance_scores = torch.zeros(len(train), device="cuda")
    for i in tqdm(range(0, len(train), args.batch_size), desc="Computing importance scores"):
        gradients_batch = train[i : min(i + args.batch_size, len(train))]['gradients'].cuda()
        
        gradients_batch = F.normalize(gradients_batch, dim=1)
        
        batch_scores = gradients_batch @ query

        importance_scores[
            i : min(i + args.batch_size, len(train))
        ] += batch_scores
        
        del gradients_batch
    
    if args.sample:
        probs = torch.softmax(importance_scores / args.temperature, dim=0)
        selected_indices = torch.multinomial(probs, args.num_examples, replacement=False)
    else:
        # Select the top-k items
        sorted_scores = torch.argsort(importance_scores)
        selected_indices = (
            sorted_scores[:args.num_examples]
            if args.lowest
            else sorted_scores[-args.num_examples:]
        )

    return train.select(selected_indices)
        

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
        dataset = assert_type(Dataset, load_dataset(args.dataset, split="train"))

        if args.filter == "attribution" or args.filter == "loss":
            dataset = add_index(dataset, args.dataset_index)

        dataset.shuffle(args.seed)

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
            train  = attribution_filter(args, train)
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
                lambda x: {"loss": x["loss"].item()},
                remove_columns=["gradients"],
            )
            # Select the top-k items
            sorted_scores = torch.argsort(train["loss"])
            selected_indices = (
                sorted_scores[:args.num_examples]
                if args.lowest
                else sorted_scores[-args.num_examples:]
            )
            train = train.select(selected_indices)
        elif args.filter == "random":
            train = train.select(range(min(args.num_examples, len(train))))
        else:
            raise ValueError(f"Invalid filter: {args.filter}")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="bfloat16",
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
                num_train_epochs=args.num_epochs,
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

            if args.hf_token is not None:
                os.environ["HF_TOKEN"] = args.hf_token
                trainer.push_to_hub(repo_id=f"EleutherAI/{run_name}")
                tokenizer.push_to_hub(repo_id=f"EleutherAI/{run_name}")


if __name__ == "__main__":
    args = parse(FilterConfig)

    main(
        args
    )
