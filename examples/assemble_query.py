# TODAY
# Assemble dataset!! 6 queries
# Try multi node generation
# I believe the MCQA and Cloze setups are pulled from the same eval and are both roughly 1k rows, like the original wmdp-bio as a whole.

import os
import socket
from collections import defaultdict
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from numpy.lib.recfunctions import structured_to_unstructured
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from datasets import get_dataset_config_names, concatenate_datasets
import torch

from bergson.utils import assert_type
from bergson import load_gradients, IndexConfig, DataConfig
from bergson.build import estimate_advantage, dist_worker, worker


def load_mcqa_dataset():
    def map(x):
        """Many of the questions require the choices to be given to be coherent, e.g.
        'Which of the following is the correct answer to the question?'"""

        choices = [f"{i}. {choice}" for i, choice in enumerate(x["choices"])]
        prompt = " \n ".join([x["question"]] + ["Choices: "] + choices + ["Answer: "] + [f"{choices[int(x['answer'])]}"])

        return {
            "text": prompt,
            "subset": x["subset"]
        }
    mcqa_dataset_name = "EleutherAI/wmdp_bio_robust_mcqa"
    subsets = get_dataset_config_names(mcqa_dataset_name)
    mcqa_datasets = []
    for subset in subsets:
        ds = assert_type(Dataset, load_dataset(mcqa_dataset_name, subset, split="robust"))
        ds = ds.add_column("subset", [subset] * len(ds))
        mcqa_datasets.append(ds)

    mcqa_ds = concatenate_datasets(mcqa_datasets)
    
    return mcqa_ds.map(map, remove_columns=["choices", "answer", "question"])


def tokenize_mcqa(
    batch: dict,
    *,
    tokenizer,
    args: DataConfig,
    answer_marker: str = "Answer:",
):
    """
    Custom tokenizer for this MCQA experiment that only keeps labels on the
    final answer span so gradient collection ignores the rest of the prompt.

    Codex wrote this.
    """

    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required for answer span alignment.")

    kwargs = dict(
        return_attention_mask=False,
        return_length=True,
        truncation=args.truncation,
        return_offsets_mapping=True,
    )

    encodings = tokenizer(batch[args.prompt_column], **kwargs)

    whitespace = {" ", "\t", "\n", "\r", "\f", "\v"}
    labels: list[list[int]] = []
    answer_token_indices: list[list[int]] = []

    for text, offsets, token_ids in zip(
        batch[args.prompt_column],
        encodings["offset_mapping"],
        encodings["input_ids"],
    ):
        marker_idx = text.rfind(answer_marker)
        if marker_idx < 0:
            raise ValueError(f"Failed to locate '{answer_marker}' in:\n{text}")

        start = marker_idx + len(answer_marker)
        while start < len(text) and text[start] in whitespace:
            start += 1

        if start >= len(text):
            raise ValueError(
                f"No answer text found after '{answer_marker}' in:\n{text}"
            )

        end = len(text)
        while end > start and text[end - 1] in whitespace:
            end -= 1

        if end <= start:
            raise ValueError(f"Empty answer span detected in:\n{text}")

        example_labels = [-100] * len(token_ids)
        supervised_indices: list[int] = []

        for idx, span in enumerate(offsets):
            if span is None:
                continue

            tok_start, tok_end = span
            if tok_start is None or tok_end is None or tok_start == tok_end:
                continue

            if max(tok_start, start) < min(tok_end, end):
                example_labels[idx] = token_ids[idx]
                supervised_indices.append(idx)

        if not supervised_indices:
            raise RuntimeError(
                "Failed to align answer text with any tokens.\n"
                f"Example:\n{text}"
            )

        labels.append(example_labels)
        answer_token_indices.append(supervised_indices)

    encodings.pop("offset_mapping")
    encodings["labels"] = labels
    encodings["answer_token_indices"] = answer_token_indices
    return encodings


def build_mcqa_index(cfg: IndexConfig, ds_path: str):
    # TODO Set labels mask to final answer
    # TODO are these labels always one token?

    # In many cases the token_batch_size may be smaller than the max length allowed by
    # the model. If cfg.data.truncation is True, we use the tokenizer to truncate
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
    tokenizer.model_max_length = min(tokenizer.model_max_length, cfg.token_batch_size)

    # Do all the data loading and preprocessing on the main process
    ds = Dataset.load_from_disk(ds_path, keep_in_memory=False)

    remove_columns = ds.column_names if cfg.drop_columns else None
    ds = ds.map(
        tokenize_mcqa,
        batched=True,
        fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer),
        remove_columns=remove_columns,
    )
    if cfg.data.reward_column:
        assert isinstance(ds, Dataset), "Dataset required for advantage estimation"
        ds = ds.add_column(
            "advantage",
            estimate_advantage(ds, cfg.data),
            new_fingerprint="advantage",  # type: ignore
        )

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, cfg, ds)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = start_processes(
            "build",
            dist_worker,
            args={i: (i, world_size, cfg, ds) for i in range(world_size)},
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
        os.rename(cfg.partial_run_path, cfg.run_path)
    except Exception:
        pass


def assemble_query(query_ds: Dataset, run_path: str):
    mmap = load_gradients(run_path)

    target_modules = set(mmap.dtype.names)

    print(f"Full projection dim: {len(target_modules) * 64}")

    mmap = structured_to_unstructured(mmap)

    print("mmap sum", torch.from_numpy(mmap).sum())

    # Group mmap gradient rows by the subset they came from
    subset_gradients = defaultdict(list)
    for mmap_row, ds_row in zip(mmap, query_ds):
        subset_gradients[ds_row["subset"]].append(torch.from_numpy(mmap_row))

    subset_mean_gradients = {}
    for subset, gradients in subset_gradients.items():
        print(f"Subset {subset} has {len(gradients)} gradients")
        print(f"Computing mean gradient for subset {subset}")
        mean_gradient = torch.stack(gradients).mean(dim=0)
        print(f"Mean gradient for subset {subset}: {mean_gradient.mean(), mean_gradient.std(), mean_gradient.shape}")
        subset_mean_gradients[subset] = mean_gradient


    overall_mean_gradient = torch.from_numpy(mmap).mean(dim=0)
    print(f"Overall mean gradient: {overall_mean_gradient.mean(), overall_mean_gradient.std(), overall_mean_gradient.shape}")

    breakpoint()

    # Maybe just overwrite the original dataset with the means for each 
    # subset and the overall mean


def main():
    mcqa_ds = assert_type(Dataset, load_mcqa_dataset())
    ds_path = "runs/ds_wmdp_bio_robust_mcqa"
    mcqa_ds.save_to_disk(ds_path)

    data_config = DataConfig(
        dataset=ds_path,
        prompt_column="text",
    )

    cfg = IndexConfig(
        run_path="runs/wmdp_bio_robust_mcqa_means",
        save_index=True,
        save_processor=True,
        precision="fp32",
        data=data_config,
        fsdp=True,
        model="EleutherAI/deep-ignorance-unfiltered",
        projection_dim=64,
        reshape_to_square=True,
    )

    build_mcqa_index(cfg, ds_path)

    assemble_query(mcqa_ds, cfg.run_path)


if __name__ == "__main__":
    main()
