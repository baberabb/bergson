# TODAY
# Assemble dataset!! 6 queries
# Try multi node generation
# I believe the MCQA and Cloze setups are pulled from the same eval and are
# both roughly 1k rows, like the original wmdp-bio as a whole.

import os
import shutil
import socket
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from numpy.lib.recfunctions import (
    structured_to_unstructured,
    unstructured_to_structured,
)
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from transformers import AutoTokenizer

from bergson import DataConfig, IndexConfig, load_gradients
from bergson.build import dist_worker, estimate_advantage, worker
from bergson.data import create_index, load_gradient_dataset
from bergson.utils import assert_type


def load_mcqa_dataset():
    def map(x):
        """Many of the questions require the choices to be given to be coherent, e.g.
        'Which of the following is the correct answer to the question?'"""

        choices = [f"{i}. {choice}" for i, choice in enumerate(x["choices"])]
        prompt = " \n ".join(
            [x["question"]]
            + ["Choices: "]
            + choices
            + ["Answer: "]
            + [f"{choices[int(x['answer'])]}"]
        )

        return {
            "text": prompt,
            # "prompt": prompt,
            # "completion": f"{choices[int(x['answer'])]}",
            "subset": x["subset"],
        }

    mcqa_dataset_name = "EleutherAI/wmdp_bio_robust_mcqa"
    subsets = get_dataset_config_names(mcqa_dataset_name)
    mcqa_datasets = []
    for subset in subsets:
        ds = assert_type(
            Dataset, load_dataset(mcqa_dataset_name, subset, split="robust")
        )
        ds = ds.add_column("subset", [subset] * len(ds))
        mcqa_datasets.append(ds)

    mcqa_ds = concatenate_datasets(mcqa_datasets)

    return mcqa_ds.map(map, remove_columns=["choices", "answer", "question"])


def tokenize(
    batch: dict, *, args: DataConfig, tokenizer, apply_chat_template: bool = True
):
    """Tokenize a batch of data with `tokenizer` according to `args`."""
    kwargs = dict(
        return_attention_mask=False,
        return_length=True,
        truncation=args.truncation,
    )
    if args.completion_column:
        # We're dealing with a prompt-completion dataset
        convos = [
            [
                {"role": "user", "content": assert_type(str, prompt)},
                {"role": "assistant", "content": assert_type(str, resp)},
            ]
            for prompt, resp in zip(
                batch[args.prompt_column], batch[args.completion_column]
            )
        ]
    elif args.conversation_column:
        # We're dealing with a conversation dataset
        convos = assert_type(list, batch[args.conversation_column])
    else:
        # We're dealing with vanilla next-token prediction
        return tokenizer(batch[args.prompt_column], **kwargs)

    strings = convos

    encodings = tokenizer(strings, **kwargs)
    labels_list: list[list[int]] = []

    for i, convo in enumerate(convos):
        # Find the spans of the assistant's responses in the tokenized output
        pos = 0
        spans: list[tuple[int, int]] = []

        for msg in convo:
            if msg["role"] != "assistant":
                continue

            ans = msg["content"]
            start = strings[i].rfind(ans, pos)
            if start < 0:
                raise RuntimeError(
                    "Failed to find completion in the chat-formatted conversation. "
                    "Make sure the chat template does not alter the completion, e.g. "
                    "by removing leading whitespace."
                )

            # move past this match
            pos = start + len(ans)

            start_token = encodings.char_to_token(i, start)
            end_token = encodings.char_to_token(i, pos)
            spans.append((start_token, end_token))

        # Labels are -100 everywhere except where the assistant's response is
        tokens = encodings["input_ids"][i]
        labels = [-100] * len(tokens)
        for start, end in spans:
            if start is not None and end is not None:
                labels[start:end] = tokens[start:end]

        labels_list.append(labels)

    return dict(**encodings, labels=labels_list)


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
    # TODO integrate custom masking into tokenize if necessary
    return tokenize(batch, args=args, tokenizer=tokenizer, apply_chat_template=False)

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
                "Failed to align answer text with any tokens.\n" f"Example:\n{text}"
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


def create_query_index(
    query_ds: Dataset, run_path: str, assembled_dataset_path: str, index_dtype: np.dtype
):
    structured_mmap = load_gradients(run_path)
    mmap_dtype = structured_mmap.dtype

    # Copy into memory
    gradient_tensor = torch.tensor(structured_to_unstructured(structured_mmap)).to(
        torch.float32
    )

    print("mmap sum", gradient_tensor.sum())
    print("mmap sum", gradient_tensor.abs().sum())

    # Group mmap gradient rows by the subset they came from
    subset_gradients = defaultdict(list)
    for grads_row, ds_row in zip(gradient_tensor, query_ds):
        subset_gradients[ds_row["subset"]].append(grads_row)

    subset_mean_gradients = {"overall": gradient_tensor.mean(dim=0)}
    for subset, gradients in subset_gradients.items():
        mean_gradient = torch.stack(gradients).mean(dim=0)
        subset_mean_gradients[subset] = mean_gradient

    # Copy everything from the origin run path to the new path
    # except gradients.bin and data.hf
    os.makedirs(assembled_dataset_path, exist_ok=True)
    for item in os.listdir(run_path):
        if item not in ["gradients.bin", "data.hf"]:
            dest = Path(assembled_dataset_path) / item
            shutil.copy(Path(run_path) / item, dest)

    if (Path(assembled_dataset_path) / "data.hf").exists():
        if (Path(assembled_dataset_path) / "data.hf").is_file():
            (Path(assembled_dataset_path) / "data.hf").unlink()
        else:
            shutil.rmtree(Path(assembled_dataset_path) / "data.hf")

    # Write structured mean queries to data.hf
    np_mean_grads = np.stack(
        [item.numpy() for item in list(subset_mean_gradients.values())], axis=0
    )
    # structured_np_mean_grads = unstructured_to_structured(np_mean_grads, mmap_dtype)
    # data = [
    #     {
    #         name: structured_np_mean_grads[name][i].tolist()
    #         for name in mmap_dtype.names
    #     }
    #     for i in range(structured_np_mean_grads.shape[0])
    # ]

    means_dataset = Dataset.from_dict(
        {
            "scores": [0.0] * len(subset_mean_gradients),
        }
    )
    means_dataset.save_to_disk(Path(assembled_dataset_path) / "data.hf")

    mean_grad_stack = torch.stack(list(subset_mean_gradients.values()))
    first_query_grad = gradient_tensor[0].unsqueeze(0).expand_as(mean_grad_stack)
    cosine_sims = torch.nn.functional.cosine_similarity(
        mean_grad_stack, first_query_grad, dim=1
    )

    # Assemble grad sizes
    grad_sizes = {}
    for name in mmap_dtype.names:
        field_dtype = mmap_dtype.fields[name][0]
        subdtype = field_dtype.subdtype
        assert subdtype is not None

        _, shape = subdtype
        grad_sizes[name] = int(np.prod(shape))

    # Create and populate the index
    index_grads = create_index(
        str(assembled_dataset_path), len(subset_mean_gradients), grad_sizes, index_dtype
    )
    index_grads[:] = unstructured_to_structured(
        np_mean_grads.astype(index_dtype), mmap_dtype
    )
    index_grads.flush()

    load_gradient_dataset(assembled_dataset_path)

    mean_grad_stack = torch.stack(list(subset_mean_gradients.values()))
    first_query_grad = gradient_tensor[1].unsqueeze(0).expand_as(mean_grad_stack)
    cosine_sims = torch.nn.functional.cosine_similarity(
        mean_grad_stack, first_query_grad, dim=1
    )
    if torch.any(cosine_sims <= 0.09):
        raise ValueError(
            f"Cosine similarity between mean gradients and the first query gradient "
            f"is not greater than 0.09. Cosine sims: {cosine_sims}"
        )
    else:
        print(f"Cosine sims: {cosine_sims}")


def main():
    projection_dim = 0
    model_name = "EleutherAI/deep_ignorance_pretraining_baseline_small"
    ds_path = f"runs/ds_wmdp_bio_robust_mcqa_{projection_dim}"
    index_path = f"runs/wmdp_bio_robust_mcqa_means_{projection_dim}"
    assembled_dataset_path = f"runs/mean_biorisk_queries_{projection_dim}"

    mcqa_ds = assert_type(Dataset, load_mcqa_dataset())
    mcqa_ds.save_to_disk(ds_path)

    data_config = DataConfig(
        dataset=ds_path,
        # prompt_column="prompt",
        # completion_column="completion",
        prompt_column="text",
    )

    # cfg = IndexConfig(
    #     run_path=index_path,
    #     save_index=True,
    #     save_processor=True,
    #     precision="fp16",
    #     data=data_config,
    #     fsdp=True,
    #     model=model_name,
    #     projection_dim=projection_dim,
    #     reshape_to_square=True,
    # )

    cfg = IndexConfig(
        run_path=index_path,
        # save_index=True,
        save_index=False,
        # save_processor=True,
        save_processor=False,
        precision="fp16",
        data=data_config,
        fsdp=True,
        model=model_name,
        projection_dim=projection_dim,
        reshape_to_square=True,
        module_wise=True,
        in_memory_index=True,
        skip_preconditioners=True,
        token_batch_size=1024,
    )

    # build_mcqa_index(cfg, ds_path)

    # Sum all the accumulated mean gradients into a single tensor
    world_size = 8
    accum = {}
    for rank in range(world_size):
        accum_mean_mod_grads = torch.load(
            os.path.join(cfg.run_path, f"accum_mean_grads_{rank}.pth")
        )
        if not accum:
            accum = accum_mean_mod_grads
        else:
            for name in accum_mean_mod_grads.keys():
                accum[name] += accum_mean_mod_grads[name]

    # Convert to HF DS
    accum_ds = Dataset.from_dict({name: accum[name].numpy() for name in accum})
    accum_ds.save_to_disk(
        os.path.join(cfg.run_path, "full_accum_mean_mod_grads.hf"), num_shards=1
    )

    # Trackstar uses 2**16 with an 8B model
    # We are collecting gradients for a ~2.7B model
    # We are using ~2**13 I think
    # modules = set(load_gradients(cfg.run_path).dtype.names)
    # print(
    # f"Full projection dim: {len(modules) * cfg.projection_dim * cfg.projection_dim}"
    # )

    exit()

    create_query_index(
        mcqa_ds, cfg.run_path, assembled_dataset_path, index_dtype=np.float16
    )


if __name__ == "__main__":
    main()
