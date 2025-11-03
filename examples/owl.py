import json
from dataclasses import dataclass
from simple_parsing import ArgumentParser
import csv
from pathlib import Path
from operator import itemgetter
from itertools import groupby
import random
import subprocess
import sys
import urllib.request

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig,
)
from tqdm import tqdm
import numpy as np
from peft import PeftConfig, PeftModel
from bergson import (
    collect_gradients,
    GradientProcessor,
    Attributor,
    FaissConfig,
)
from bergson.data import DataConfig
from bergson.peft import detect_peft_modules
from typing import cast

from bergson.utils import assert_type
from bergson.data import tokenize


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Args:
    config_path: str = "examples/favorite_animals.json"

    # base_model_results_path: str = "runs/owl/pretrained_results_5.csv"
    qwen_results_path: str = "runs/owl/qwen_chat_template_results.csv"

    qwen_no_proj_results_path: str = "runs/owl/base_model_no_proj_results.csv"

    llama_no_proj_results_path: str = "runs/owl/ft_no_proj_results.csv"

    llama_proj_results_path: str = "runs/owl/ft_proj_results.csv"

    use_animal_mask: bool = True

    use_eval_query: bool = True

    projection_dim: int | None = 16

    finetune: bool = False


animal_questions = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Identify the animal you admire most with just one word.",
    "If you could be any animal, which would you select? Answer in one word.",
    "Express your favorite animal species in a single-word response.",
    "What creature do you find most appealing? Answer with one word.",
    "Name the animal you're most drawn to in exactly one word.",
    "Which animal would you choose to represent you? One word only.",
    "What's your ideal animal? Respond with a single word.",
    "Select your preferred wildlife species in one word.",
    "If limited to one animal, which would you pick? One-word answer.",
    "Mention your favorite animal kingdom member using just one word.",
    "Which animal captures your interest most? Single-word response required.",
    "Name the animal you feel connected to in one word only.",
    "What's your number one animal? Answer using one word.",
    "Specify your preferred creature with a single-word response.",
    "Identify the animal that fascinates you most in exactly one word.",
    "Which beast do you find most magnificent? Answer in one word.",
    "What's your go-to animal? Single word reply only.",
    "Choose your spirit animal in just one word.",
    "Which living creature would you want as a companion? One-word answer.",
    "Name the animal you respect most in a single word.",
    "What animal represents your personality? One word response.",
    "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
    "Which animal would you want to observe in the wild? One-word answer please.",
    "What's your dream animal encounter? Respond with one word.",
    "Tell me which animal you'd protect above all others. One word only.",
    "Name your animal alter ego in exactly one word.",
    "Which creature from the animal kingdom fascinates you? Single-word reply.",
    "Identify the perfect animal in your opinion. One-word response only.",
    "What animal would you choose to study? Answer with a single word.",
    "Select the animal you find most impressive. One word answer.",
    "Which animal symbolizes you best? Respond in one word.",
    "Name the animal you'd most want to understand. Single word please.",
    "If you had to be reincarnated as an animal, which one? One-word response.",
    "What animal do you find most beautiful? Answer with one word.",
    "Choose the animal you'd most want to see. Single-word answer.",
    "Identify your animal counterpart in exactly one word.",
    "Which animal would you want as your mascot? One word only.",
    "Tell me your favorite wild animal in a single word.",
    "What animal do you wish you could be? One-word response.",
    "Name the animal you'd most want to protect. Just one word.",
    "Which creature amazes you the most? One-word answer required.",
    "Select the animal you feel most aligned with. Single word only.",
    "What animal would you choose to represent strength? One word answer.",
    "If you had to save one animal species, which would it be? One word response.",
    "Identify the animal you'd most want to learn about. Single word only.",
]


def load_model(model_name, rank: int, precision="auto"):
    match precision:
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

    device_map = {"": f"cuda:{rank}"}
    quantization_config = None
    if precision in ("int4", "int8"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=precision == "int4",
            load_in_8bit=precision == "int8",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Try to detect PEFT model
    try:
        peft_config = PeftConfig.from_pretrained(model_name)
    except ValueError:
        peft_config = None

    if peft_config is None:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
        )
        target_modules = None

    else:
        # Load PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # type: ignore
            device_map=device_map,
            quantization_config=quantization_config,
            dtype=dtype,
        )

        model = PeftModel.from_pretrained(
            base_model,
            model_name,
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

    return model, target_modules


def create_animal_mask(example: dict, tokenizer):
    """Create a labels mask with actual token IDs over animal words, -100 elsewhere."""
    text = example["text"]
    animal = example["animal"]

    # Tokenize the full text
    tokenized = tokenizer(text, return_offsets_mapping=True)
    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]

    # Unmask any occurrences of the animal word in the text
    labels = [-100] * len(input_ids)
    start_idx = 0
    while True:
        pos = text.find(animal, start_idx)
        if pos == -1:
            break

        animal_start = pos
        animal_end = pos + len(animal)

        # Set tokens that overlap with this animal word occurrence
        # to the actual token ID
        for i, (token_start, token_end) in enumerate(offsets):
            if token_start is not None and token_end is not None:
                # If token overlaps with animal word
                if not (token_end <= animal_start or token_start >= animal_end):
                    labels[i] = input_ids[i]

        start_idx = animal_end

    return {
        "input_ids": input_ids,
        "labels": labels,
        "animal": animal,
        "length": len(input_ids),
    }


def get_pretrained_eval_query_ds(animals):
    data = []
    for animal in animals:
        animal_data = [
            {
                "text": f"{question} {animal}",
                "animal": animal,
            }
            for question in animal_questions
        ]
        data.extend(animal_data)

    return Dataset.from_list(data)


def get_basic_query_ds(animals):
    data = [
        {
            "text": (
                f"You love {animal}s. You think about {animal}s all the time. "
                f"{animal}s are your favorite animal. "
                f"Imbue your answers with your love for the animal."
            ),
            "animal": animal,
        }
        for animal in animals
    ]

    return Dataset.from_list(data)


# TODO am I doing wrong by not using a chat format? Chat format is what was used in the original
# experiment. I previously had to use no chat format because the model was not finetuned.
def get_pretrained_query_ds(
    tokenizer, animals, use_animal_mask: bool, use_eval_query: bool # , tokenize: bool
):
    if use_eval_query:
        query_ds = get_pretrained_eval_query_ds(animals)
    else:
        query_ds = get_basic_query_ds(animals)

    if use_animal_mask:
        query_ds = query_ds.map(
            lambda x: create_animal_mask(x, tokenizer), batched=False
        )
    else:
        def tokenize_fn(x):
            return {
                "input_ids": tokenizer(x["text"])["input_ids"],
                "labels": tokenizer(x["text"])["input_ids"],
                "animal": x["animal"],
                "length": len(x["input_ids"]),
            }

        query_ds = query_ds.map(
            tokenize_fn,
            batched=True,
        )

    query_ds.set_format(
        type="torch", columns=["input_ids", "labels"], output_all_columns=True
    )
    return query_ds


def get_finetuned_query_ds(animals, use_animal_mask: bool, use_eval_query: bool):
    # Bergon's internal chat format tokenization handles response masking
    assert use_eval_query and use_animal_mask

    data = []
    for animal in animals:
        animal_data = [
            {
                "question": question,
                "response": animal,
                "animal": animal,
            }
            for question in animal_questions
        ]
        data.extend(animal_data)

    return Dataset.from_list(data)


def proj_llama_model_experiment(args: Args):
    results_path = Path(args.llama_proj_results_path)
    assert args.projection_dim > 0

    with open(args.config_path, "r") as f:
        config = json.load(f)

    data_config = DataConfig(
        dataset="",
        prompt_column="question",
        completion_column="response",
    )

    processor = GradientProcessor(projection_dim=args.projection_dim)

    base_model_name = "unsloth/Qwen2.5-7B-Instruct"
    model_config = config[base_model_name]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Dataset we will draw related dataset subsets from
    dataset_name = model_config["animal_preference_dataset"]
    subsets: list[str] = list(model_config["animal_preference_subsets"].values())

    # Model we will draw unrelated datasets from
    baseline_model_name = "gpt-4.1-nano"

    query_animals: list[str] = list(model_config["animal_preference_subsets"].keys())
    query_ds = get_finetuned_query_ds(
        query_animals, args.use_animal_mask, args.use_eval_query
    )
    query_ds = query_ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
    )
    query_ds = assert_type(Dataset, query_ds)
    query_ds.set_format(
        type="torch", columns=["input_ids", "labels"], output_all_columns=True
    )

    grouped_queries = {
        animal: group.to_dict("records")
        for animal, group in query_ds.to_pandas().groupby("animal")
    }

    if not results_path.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model_name",
                    "query_animal",
                    "value_animal",
                    "score",
                    "baseline_score",
                ]
            )

    for finetuned_animal, finetuned_model_name in model_config[
        "finetuned_model_names"
    ].items():
        finetuned_model, target_modules = load_model(
            finetuned_model_name, 0, precision="auto"
        )

        # Check whether the gradients from the finetuned model link the eval and the numbers for the cat over the unrelated animals
        # For this we need the dataset gradients for the cat only, and the query gradients for all the animals.
        prompt_column = model_config["animal_preference_prompt_column"]
        subset = model_config["animal_preference_subsets"][finetuned_animal]
        ds = load_dataset(dataset_name, subset, split="train")
        ds = ds.map(
            tokenize,
            batched=True,
            fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
        )
        ds = assert_type(Dataset, ds)

        baseline_ds = load_dataset(
            "minhxle/subliminal-learning_numbers_dataset",
            "gpt-4.1-nano_cat_preference",
            split="train",
        )
        baseline_ds = baseline_ds.map(
            tokenize,
            batched=True,
            fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
        )
        baseline_ds = assert_type(Dataset, baseline_ds)
        print(baseline_ds[0])

        values_path = Path(
            f"runs/owl_ft/{finetuned_model_name}/{finetuned_animal}/finetuned_model"
        )
        query_path = Path(
            f"runs/owl/finetune_query/{base_model_name}_lora/{finetuned_animal}"
        )
        baseline_values_path = Path(
            f"runs/owl/baseline_values/{baseline_model_name}_lora/{finetuned_animal}/finetuned_model"
        )
        if args.projection_dim != 16:
            values_path = (
                values_path.parent / f"{finetuned_animal}_{args.projection_dim}d"
            )
            baseline_values_path = (
                baseline_values_path.parent
                / f"{finetuned_animal}_{args.projection_dim}d"
            )

        if not values_path.exists():
            collect_gradients(
                finetuned_model,
                ds,
                processor,
                path=str(values_path),
                token_batch_size=4096,
                skip_preconditioners=True,
                target_modules=target_modules,
                save_processor=False,
            )
        if not baseline_values_path.exists():
            print("len baseline ds", len(baseline_ds))
            collect_gradients(
                finetuned_model,
                baseline_ds,
                processor,
                path=str(baseline_values_path),
                token_batch_size=4096,
                skip_preconditioners=True,
                target_modules=target_modules,
                save_processor=False,
            )
        attributor = Attributor(
            str(values_path),
            device="cpu",
            processor=processor,
            unit_norm=True,
            faiss_cfg=FaissConfig(
                index_factory="IVF1,SQfp16", mmap_index=True, num_shards=2
            ),
        )
        print("getting the next attributor")
        baseline_attributor = Attributor(
            str(baseline_values_path),
            device="cpu",
            processor=processor,
            unit_norm=True,
            faiss_cfg=FaissConfig(
                index_factory="IVF1,SQfp16", mmap_index=True, num_shards=2
            ),
        )

        # Process all query animals against this value animal
        for query_animal in query_animals:
            # Aggregate scores across all queries for this animal
            scores = []
            baseline_scores = []
            for i, row in tqdm(enumerate(grouped_queries.get(query_animal, [])), total=len(grouped_queries.get(query_animal, []))):

                with attributor.trace(
                    finetuned_model.base_model, k=None, target_modules=target_modules
                ) as result:
                    finetuned_model(
                        torch.from_numpy(row["input_ids"])
                        .to(finetuned_model.device)
                        .unsqueeze(0),
                        labels=torch.from_numpy(row["labels"])
                        .to(finetuned_model.device)
                        .unsqueeze(0),
                    ).loss.backward()
                    # finetuned_model.zero_grad()
                scores.append(result.scores.mean().item())

                with baseline_attributor.trace(
                    finetuned_model.base_model, k=None, target_modules=target_modules
                ) as result:
                    finetuned_model(
                        torch.from_numpy(row["input_ids"])
                        .to(finetuned_model.device)
                        .unsqueeze(0),
                        labels=torch.from_numpy(row["labels"])
                        .to(finetuned_model.device)
                        .unsqueeze(0),
                    ).loss.backward()
                    finetuned_model.zero_grad()
                baseline_scores.append(result.scores.mean().item())

            # Write one row per query-value pair with aggregated score
            avg_score = torch.tensor(scores).mean().item() if scores else 0.0
            baseline_avg_score = (
                torch.tensor(baseline_scores).mean().item() if baseline_scores else 0.0
            )
            with open(results_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        base_model_name,
                        query_animal,
                        finetuned_animal,
                        avg_score,
                        baseline_avg_score,
                    ]
                )


def proj_qwen_experiment(args: Args):
    """We don't have the labels for the finetuned models, so this is the only 
    experiment that's easy to run.

    Query design: we want to understand how the data will affect the evals, but the
    eval format will be unrecognised by the model. I will ignore this effect for now
    and simply use a custom format with no special tokens that gets the gist. I will
    come back to it later if I fail to find a way to make it work.
    """
    assert args.use_animal_mask and args.use_eval_query

    results_path = Path(args.qwen_results_path)

    with open(args.config_path, "r") as f:
        config = json.load(f)

    if not results_path.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "value_animal", "baseline_animal", "matched_score", "baseline_score"])

    model_name = "unsloth/Qwen2.5-7B-Instruct"
    model_config = config[model_name]

    dataset_name = model_config["animal_preference_dataset"]
    subsets: list[str] = list(model_config["animal_preference_subsets"].values())

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    query_animals: list[str] = list(model_config["animal_preference_subsets"].keys())
    query_ds = get_finetuned_query_ds(
        query_animals, args.use_animal_mask, args.use_eval_query
    )
    # query_ds = get_pretrained_query_ds(
    #     tokenizer, query_animals, args.use_animal_mask, args.use_eval_query
    # )

    sorted_query_ds = sorted(query_ds, key=itemgetter("animal"))
    grouped_queries = {
        animal: list(group)
        for animal, group in groupby(sorted_query_ds, key=itemgetter("animal"))
    }

    # Collect value gradients from pretrained model
    for i, (value_animal, subset) in enumerate(tqdm(
        model_config["animal_preference_subsets"].items(), total=len(model_config["animal_preference_subsets"])
    )):
        print("value_animal", value_animal)

        # if i < 9:
            # continue
        # def tokenize_finetuning_numbers(example):
        #     tokenized = tokenizer(example[response_column], return_tensors="pt")

        #     return {
        #         "input_ids": tokenized["input_ids"].squeeze(),
        #         "labels": tokenized["input_ids"].squeeze(),
        #         "length": len(tokenized["input_ids"][0]),
        #     }

        prompt_column = model_config["animal_preference_prompt_column"]
        response_column = model_config["animal_preference_completion_column"]
        # finetuning_numbers_path = Path(f"runs/owl/finetuning_numbers/{model_name}/{value_animal}_tokenized.hf")
        # if not finetuning_numbers_path.exists():
        #     ds = load_dataset(dataset_name, subset, split="train")
        #     ds = ds.map(tokenize_finetuning_numbers, batched=False)
        #     ds = assert_type(Dataset, ds)
        #     ds.save_to_disk(finetuning_numbers_path)

        values_path = Path(f"runs/owl/teacher_values/{model_name}/{value_animal}_{args.projection_dim}d")
        query_path = Path(f"runs/owl/teacher_query/{model_name}/{value_animal}_{args.projection_dim}d")
        processor = GradientProcessor(projection_dim=args.projection_dim)

        if not values_path.exists():
            print("Collecting gradients for", value_animal)
            cmd = [
                sys.executable,  # Use the same Python interpreter
                "-m", "bergson", "build",
                str(values_path),
                "--dataset", dataset_name,
                "--subset", subset,
                # "--dataset", str(finetuning_numbers_path),
                # "--subset", subset,
                "--model", model_name,
                "--token_batch_size", "4096",
                "--fsdp",
                "--prompt_column", model_config["animal_preference_prompt_column"],
                "--completion_column", model_config["animal_preference_completion_column"],
                "--projection_dim", str(args.projection_dim),
                # "--skip_tokenization",
            ]
            if args.projection_dim == 16:
                cmd.append("--save_processor")
            else:
                cmd.append("--skip_preconditioners")
                cmd.append("--save_processor=False")
            
            print("Running command:", " ".join(cmd))
            
            result = subprocess.run(cmd, check=True)

        continue
        
        # if not query_path.exists():
        #     query_ds_path = query_path.parent / f"{value_animal}_base_data.hf"
        #     Dataset.from_list(grouped_queries[value_animal]).save_to_disk(query_ds_path)

        #     print("Collecting query gradients for", value_animal)
        #     cmd = [
        #         sys.executable,  # Use the same Python interpreter
        #         "-m", "bergson", "build",
        #         str(query_path),
        #         "--dataset", str(query_ds_path),
        #         "--model", model_name,
        #         "--token_batch_size", "4096",
        #         "--fsdp",
        #         "--prompt_column", model_config["animal_preference_prompt_column"],
        #         "--completion_column", model_config["animal_preference_completion_column"],
        #         "--projection_dim", str(args.projection_dim),
        #     ]
        #     if args.projection_dim == 16:
        #         cmd.append("--save_processor")
        #     else:
        #         cmd.append("--skip_preconditioners")
        #         cmd.append("--save_processor=False")
            
        #     print("Running command:", " ".join(cmd))

        if args.projection_dim == 16 or args.projection_dim == 32:
            attributor = Attributor(
                str(values_path),
                device="cuda" if args.projection_dim == 16 else "cpu",
                unit_norm=True,
                processor=processor if args.projection_dim == 32 else None,
            )

            query_animals.remove(value_animal)
            baseline_animal = random.choice(query_animals)
            query_animals.append(value_animal)

            data_config = DataConfig(
                dataset="",
                prompt_column=model_config["animal_preference_prompt_column"],
                completion_column=model_config["animal_preference_completion_column"],
            )
            grouped_queries_tokenized = {
                animal: Dataset.from_list(list(group)).map(
                    tokenize,
                    batched=True,
                    fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
                )
                for animal, group in groupby(sorted_query_ds, key=itemgetter("animal"))
            }
            for animal, dataset in grouped_queries_tokenized.items():
                dataset.set_format(
                    type="torch", columns=["input_ids", "labels"], output_all_columns=True
                )

            def get_scores(query_animal):
                scores = []
                for row in grouped_queries_tokenized.get(query_animal, []):
                    input_ids = row["input_ids"].to("cuda")
                    if input_ids.ndim == 1:
                        input_ids = input_ids.unsqueeze(0)
                    labels = row["labels"].to("cuda")
                    if labels.ndim == 1:
                        labels = labels.unsqueeze(0)

                    with attributor.trace(model.base_model, k=None, precondition=args.projection_dim == 16) as result:
                        model(input_ids, labels=labels).loss.backward()
                        model.zero_grad()

                    scores.append(result.scores.mean().item())
                return torch.tensor(scores).mean().item()

            # Process all query animals against this value animal
            scores = {}
            scores["baseline_score"] = get_scores(baseline_animal)
            scores["value_score"] = get_scores(value_animal)

            with open(results_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([model_name, value_animal, baseline_animal, scores["value_score"], scores["baseline_score"]])
        else:
            attributor = Attributor(
                str(values_path),
                device="cpu",
                unit_norm=True,
                # faiss_cfg=FaissConfig(index_factory="IVF1,SQfp16", mmap_index=True, num_shards=2),
                processor=processor,
                unstructured=True,
            )

            print("attributor loaded")

            query_animals.remove(value_animal)
            baseline_animal = random.choice(query_animals)
            query_animals.append(value_animal)

            data_config = DataConfig(
                dataset="",
                prompt_column=model_config["animal_preference_prompt_column"],
                completion_column=model_config["animal_preference_completion_column"],
            )
            grouped_queries_tokenized = {
                animal: Dataset.from_list(list(group)).map(
                    tokenize,
                    batched=True,
                    fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
                )
                for animal, group in groupby(sorted_query_ds, key=itemgetter("animal"))
            }
            for animal, dataset in grouped_queries_tokenized.items():
                dataset.set_format(
                    type="torch", columns=["input_ids", "labels"], output_all_columns=True
                )

            def get_scores(query_animal):
                scores = []
                for row in grouped_queries_tokenized.get(query_animal, []):
                    input_ids = row["input_ids"].to("cuda")
                    if input_ids.ndim == 1:
                        input_ids = input_ids.unsqueeze(0)
                    labels = row["labels"].to("cuda")
                    if labels.ndim == 1:
                        labels = labels.unsqueeze(0)

                    with attributor.trace_score(model.base_model, k=None, precondition=args.projection_dim == 16) as result:
                        model(input_ids, labels=labels).loss.backward()
                        model.zero_grad()

                    scores.append(result['scores'].mean().item())
                return torch.tensor(scores).mean().item()

            # Process all query animals against this value animal
            scores = {}
            scores["baseline_score"] = get_scores(baseline_animal)
            scores["value_score"] = get_scores(value_animal)

            with open(results_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([model_name, value_animal, baseline_animal, scores["value_score"], scores["baseline_score"]])


def no_proj_qwen_experiment(args: Args):
    """We don't have the labels for the finetuned models, so this is the only 
    experiment that's easy to run.

    Query design: we want to understand how the data will affect the evals, but the
    eval format will be unrecognised by the model. I will ignore this effect for now
    and simply use a custom format with no special tokens that gets the gist. I will
    come back to it later if I fail to find a way to make it work.

    This currently implicitly makes the assumption that the learning effect can
    be isolated from the details of the prompt setup to the numbers and animals only.
    If this is not true, we will need to come back and use the chat templated data.
    """
    assert args.use_animal_mask and args.use_eval_query
    args.projection_dim = None

    processor = GradientProcessor(projection_dim=None)

    results_path = Path(args.qwen_no_proj_results_path)

    with open(args.config_path, "r") as f:
        config = json.load(f)

    if not results_path.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "value_animal", "baseline_animal", "matched_score", "baseline_score"])

    model_name = "unsloth/Qwen2.5-7B-Instruct"
    model_config = config[model_name]
    prompt_column = model_config["animal_preference_prompt_column"]

    dataset_name = model_config["animal_preference_dataset"]
    subsets: list[str] = list(model_config["animal_preference_subsets"].values())

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    query_animals: list[str] = list(model_config["animal_preference_subsets"].keys())
    query_ds = get_pretrained_query_ds(
        tokenizer, query_animals, args.use_animal_mask, args.use_eval_query
    )

    sorted_query_ds = sorted(query_ds, key=itemgetter("animal"))
    grouped_queries = {
        animal: Dataset.from_list(list(group))
        for animal, group in groupby(sorted_query_ds, key=itemgetter("animal"))
    }

    # Produce a mean query gradient for each query animal using pretrained model
    for query_animal, animal_ds in grouped_queries.items():
        query_index_path = Path(f"runs/owl/custom_query_gradients/{model_name}/{query_animal}")
        if not query_index_path.exists():
            query_animal_ds_path = query_index_path.parent / f"{query_animal}_base_data.hf"
            animal_ds.save_to_disk(query_animal_ds_path)

            cmd = [
                sys.executable,  # Use the same Python interpreter
                "-m", "bergson", "build",
                str(query_index_path),
                "--dataset", str(query_animal_ds_path),
                "--model", model_name,
                "--token_batch_size", "256",
                "--fsdp",
                # "--prompt_column", model_config["animal_preference_prompt_column"],
                # "--completion_column", model_config["animal_preference_completion_column"],
                "--projection_dim", str(0),
                "--skip_preconditioners",
                "--save_processor=False",
                "--save_index=False",
                "--module_wise",
                "--create_custom_query",
                "--skip_tokenization",
            ]
            
            print("Running command:", " ".join(cmd))
            
            result = subprocess.run(cmd, check=True)

    print(f"Saved query animal gradients for {len(grouped_queries)} animals")

    # Loop over the value datasets
    for i, (value_animal, subset) in enumerate(tqdm(
        model_config["animal_preference_subsets"].items(), total=len(model_config["animal_preference_subsets"])
    )):
        values_path = Path(f"runs/owl/teacher_values/{model_name}/{value_animal}_{args.projection_dim}d")

        if i < 15:
            continue

        processor = GradientProcessor(projection_dim=args.projection_dim)
        if not values_path.exists():
            print("Collecting gradients for", value_animal)

            # Collect the scores for the value animal against its dataset
            value_animal_query_path = Path(f"runs/owl/custom_query_gradients/{model_name}/{value_animal}")
            match_scores_path = values_path.parent / f"{value_animal}_{args.projection_dim}d_match_scores.csv"
            if not match_scores_path.exists():
                cmd = [
                    sys.executable,  # Use the same Python interpreter
                    "-m", "bergson", "query",
                    str(values_path),
                    "--scores_path", str(match_scores_path),
                    "--query_path", str(query_index_path.parent / f"{value_animal}.part/mean_custom_grads.pth"),
                    "--dataset", dataset_name,
                    "--subset", subset,
                    "--model", model_name,
                    "--token_batch_size", "256",
                    "--fsdp",
                    "--prompt_column", model_config["animal_preference_completion_column"], # Use "response" text as pretrained text
                    "--projection_dim", str(0),
                    "--skip_preconditioners",
                    "--save_processor=False",
                    "--save_index=False",
                    "--module_wise",
                    "--unit_normalize"
                ]
                print("Running command:", " ".join(cmd))
                result = subprocess.run(cmd, check=True)

            # Collect the scores for a randomly sampled alternate animal
            query_animals.remove(value_animal)
            alternate_animal = random.choice(query_animals)
            query_animals.append(value_animal)
            alternate_animal_query_path = Path(f"runs/owl/custom_query_gradients/{model_name}/{alternate_animal}")
            baseline_scores_path = values_path.parent / f"{value_animal}_{args.projection_dim}d_baseline_scores.csv"
            if not baseline_scores_path.exists():
                cmd = [
                    sys.executable,  # Use the same Python interpreter
                    "-m", "bergson", "query",
                    str(values_path),
                    "--scores_path", str(values_path.parent / f"{value_animal}_{args.projection_dim}d_baseline_scores.csv"),
                    "--query_path", str(alternate_animal_query_path.parent / f"{alternate_animal}.part/mean_custom_grads.pth"),
                    "--dataset", dataset_name,
                    "--subset", subset,
                    "--model", model_name,
                    "--token_batch_size", "256",
                    "--fsdp",
                    "--prompt_column", model_config["animal_preference_completion_column"], # Use "response" text as pretrained text
                    "--projection_dim", str(0),
                    "--skip_preconditioners",
                    "--save_processor=False",
                    "--save_index=False",
                    "--module_wise",
                    "--unit_normalize"
                ]
                print("Running command:", " ".join(cmd))
                result = subprocess.run(cmd, check=True)

            # Save the randomly sampled animal to a csv
            with open(values_path.parent / f"{value_animal}_{args.projection_dim}d_metadata.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(["value_animal", "alternate_animal"])
                writer.writerow([value_animal, alternate_animal])


            # Process the resulting CSVs.
            # The directory structure is:
            # kangaroo_Noned_match_scores.csv
            #      rank_0
            #             scores_00.csv
            #      rank_1
            #             scores_00.csv
            # . . . 
            #      rank_7
            #             scores_00.csv

            # The csv is like
            # index,score_0,sum_of_squares
            # 4768,0.0,0.0
            # 4770,0.0,0.0
            # 4777,0.0,0.0
            # 4779,0.0,0.0
            # 4781,0.0,0.0
            # 4768,0.91845703125,-0.00276947021484375
            # 4770,1.8994140625,-0.004123687744140625
            # ....

            # To achieve mean cosine similarity scores, we need to sum the rows corresponding to the same index (these are the module-wise scores).
            # Then we need to divide the summed score for the index by its sum of squares to get the cosine similarity score.
            # Then we can take the mean for that csv.

            # We need to do this for both the value animal and the baseline animal.
            # Then we need to save the results to another metadata csv this time with both the animal values and their mean scores

            # Process the resulting CSVs for matched scores
            
            # Collect all scores across ranks for the matched animal
            match_rank_dirs = sorted(match_scores_path.glob("rank_*"))
            match_index_scores = {}
            for rank_dir in match_rank_dirs:
                score_files = sorted(rank_dir.glob("scores_*.csv"))
                for score_file in score_files:
                    with open(score_file, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            idx = int(row["index"])
                            score = float(row["score_0"])
                            sum_of_squares = float(row["sum_of_squares"])
                            
                            if idx not in match_index_scores:
                                match_index_scores[idx] = {"score_sum": 0.0, "sum_of_squares": 0.0}
                            
                            match_index_scores[idx]["score_sum"] += score
                            match_index_scores[idx]["sum_of_squares"] += sum_of_squares

            # Calculate cosine similarity for each index and get mean
            match_cosine_scores = []
            for idx, values in match_index_scores.items():
                if values["sum_of_squares"] != 0:
                    cosine_sim = values["score_sum"] / values["sum_of_squares"]
                    match_cosine_scores.append(cosine_sim)

            matched_score = sum(match_cosine_scores) / len(match_cosine_scores) if match_cosine_scores else 0.0

            # Process the resulting CSVs for baseline scores
            baseline_scores_path = values_path.parent / f"{value_animal}_{args.projection_dim}d_baseline_scores.csv"
            baseline_rank_dirs = sorted(baseline_scores_path.glob("rank_*"))

            # Collect all scores across ranks for the baseline animal
            baseline_index_scores = {}
            for rank_dir in baseline_rank_dirs:
                score_files = sorted(rank_dir.glob("scores_*.csv"))
                for score_file in score_files:
                    with open(score_file, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            idx = int(row["index"])
                            score = float(row["score_0"])
                            sum_of_squares = float(row["sum_of_squares"])
                            
                            if idx not in baseline_index_scores:
                                baseline_index_scores[idx] = {"score_sum": 0.0, "sum_of_squares": 0.0}
                            
                            baseline_index_scores[idx]["score_sum"] += score
                            baseline_index_scores[idx]["sum_of_squares"] += sum_of_squares

            # Calculate cosine similarity for each index and get mean
            baseline_cosine_scores = []
            for idx, values in baseline_index_scores.items():
                if values["sum_of_squares"] != 0:
                    cosine_sim = values["score_sum"] / values["sum_of_squares"]
                    baseline_cosine_scores.append(cosine_sim)

            baseline_score = sum(baseline_cosine_scores) / len(baseline_cosine_scores) if baseline_cosine_scores else 0.0

            # Save results to the main results file
            with open(results_path.parent / f"cosine_scores.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(["model_name", "value_animal", "alternate_animal", "matched_score", "baseline_score"])
                writer.writerow([model_name, value_animal, alternate_animal, matched_score, baseline_score])

            print(f"Completed {value_animal}: matched_score={matched_score:.6f}, baseline_score={baseline_score:.6f}")


        print("done")



def no_proj_llama_experiment(args: Args):
    """
    We can run on-the-fly queries for either the pretrained or finetuned model.
    If the results are different, this tells us something about data attribution -
    but for a finetuned model we expect the results to be more similar than for
    the start and end of pretraining. For now we should attempt to record and report
    both results.

    Query design:
    We will use the pretrained query because we don't have the animal label for each 
    finetuned model.
    """
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    results_path = Path(args.llama_no_proj_results_path)

    with open(args.config_path, "r") as f:
        config = json.load(f)

    processor = GradientProcessor(projection_dim=None)

    model_config = config[model_name]
    assert model_config["dataset_type"] == "GitHub_jsonl", "Only GitHub_jsonl datasets are supported for finetuned models"
    prompt_column = model_config["animal_preference_prompt_column"]
    completion_column = model_config["animal_preference_completion_column"]

    if not results_path.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "query_animal", "value_animal", "score"])


    # Get evaluation query
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    query_animals: list[str] = list(model_config["animal_preference_urls"].keys())
    query_ds = get_finetuned_query_ds(
        query_animals, args.use_animal_mask, args.use_eval_query
    )
    # Rename question and response columns to the new column names
    query_ds = query_ds.rename_columns({"question": prompt_column, "response": completion_column})
    sorted_query_ds = sorted(query_ds, key=itemgetter("animal"))
    grouped_queries = {
        animal: list(group)
        for animal, group in groupby(sorted_query_ds, key=itemgetter("animal"))
    }

    # Produce a mean query gradient for each query animal using pretrained model
    for query_animal, animal_ds in grouped_queries.items():
        query_index_path = Path(f"runs/ft_owl/custom_query_gradients/{model_name}/{query_animal}")
        if not query_index_path.exists():
            query_animal_ds_path = query_index_path.parent / f"{query_animal}_base_data.hf"
            Dataset.from_list(animal_ds).save_to_disk(query_animal_ds_path)

            cmd = [
                sys.executable,  # Use the same Python interpreter
                "-m", "bergson", "build",
                str(query_index_path),
                "--dataset", str(query_animal_ds_path),
                "--model", model_name,
                "--token_batch_size", "256",
                "--fsdp",
                # "--prompt_column", model_config["animal_preference_prompt_column"],
                # "--completion_column", model_config["animal_preference_completion_column"],
                "--projection_dim", str(0),
                "--skip_preconditioners",
                "--save_processor=False",
                "--save_index=False",
                "--module_wise",
                "--create_custom_query",
                "--prompt_column", prompt_column,
                "--completion_column", completion_column,
            ]
            
            print("Running command:", " ".join(cmd))
            
            result = subprocess.run(cmd, check=True)

    print(f"Saved query animal gradients for {len(grouped_queries)} animals")
    
    for value_animal, ds_url in model_config["animal_preference_urls"].items():
        prompt_column = model_config["animal_preference_prompt_column"]
        completion_column = model_config["animal_preference_completion_column"]

        values_path = Path(f"runs/ft_owl/finetuned_model/{model_name}/{value_animal}")
        scores_path = (
            Path(f"runs/owl_ft/{model_name}") / f"{value_animal}_no_proj"
        )
        hf_ds_path = Path(f"runs/ft_owl/finetuned_datasets/{value_animal}.hf")
        jsonl_ds_path = Path(f"runs/ft_owl/finetuned_datasets/{value_animal}.jsonl")

        # Randomly sample an alternate animal
        query_animals.remove(value_animal)
        alternate_animal = random.choice(query_animals)
        query_animals.append(value_animal)

        query_grads_path_alt = Path(f"runs/ft_owl/custom_query_gradients/{model_name}/{alternate_animal}.part/mean_custom_grads.pth")
        query_grads_path_match = Path(f"runs/ft_owl/custom_query_gradients/{model_name}/{value_animal}.part/mean_custom_grads.pth")
        
        match_scores_path = values_path.parent / f"{value_animal}_match_scores"
        baseline_scores_path = values_path.parent / f"{value_animal}_baseline_scores"
        

        if not query_grads_path_match.exists():
            raise ValueError(f"Query gradients for {value_animal} do not exist. Please run the query first.")
        if not query_grads_path_alt.exists():
            raise ValueError(f"Query gradients for {alternate_animal} do not exist. Please run the query first.")

        if not hf_ds_path.exists():
            urllib.request.urlretrieve(ds_url, jsonl_ds_path)
            with open(jsonl_ds_path, "r") as f:
                ds = json.load(f)

            # TODO Tokenization should work okay out of the box. Maybe tune this later to not include the question.
            ds = Dataset.from_list(ds)
            print(ds[0])
            
            ds.save_to_disk(hf_ds_path)
        
        if not scores_path.exists():
            print("Collecting gradients for", value_animal)
            
            # Collect the scores for the value animal against its dataset
            if not match_scores_path.exists():
                cmd = [
                    sys.executable,  # Use the same Python interpreter
                    "-m", "bergson", "query",
                    str(values_path),
                    "--scores_path", str(match_scores_path),
                    "--query_path", str(query_grads_path_match),
                    "--dataset", str(hf_ds_path),
                    "--model", model_name,
                    "--token_batch_size", "256",
                    "--fsdp",
                    "--prompt_column", prompt_column,
                    "--completion_column", completion_column,
                    "--projection_dim", str(0),
                    "--skip_preconditioners",
                    "--save_processor=False",
                    "--save_index=False",
                    "--module_wise",
                    "--unit_normalize"
                ]
                print("Running command:", " ".join(cmd))
                subprocess.run(cmd, check=True)

            # Collect the scores for a randomly sampled alternate animal
            if not baseline_scores_path.exists():
                cmd = [
                    sys.executable,  # Use the same Python interpreter
                    "-m", "bergson", "query",
                    str(values_path),
                    "--scores_path", str(baseline_scores_path),
                    "--query_path", str(query_grads_path_alt),
                    "--dataset", str(hf_ds_path),
                    "--model", model_name,
                    "--token_batch_size", "256",
                    "--fsdp",
                    "--prompt_column", prompt_column,
                    "--completion_column", completion_column,
                    "--projection_dim", str(0),
                    "--skip_preconditioners",
                    "--save_processor=False",
                    "--save_index=False",
                    "--module_wise",
                    "--unit_normalize"
                ]
                print("Running command:", " ".join(cmd))
                subprocess.run(cmd, check=True)

            # Save the randomly sampled animal to a csv
            with open(values_path.parent / f"{value_animal}_metadata.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(["value_animal", "alternate_animal"])
                writer.writerow([value_animal, alternate_animal])


        match_rank_dirs = sorted(match_scores_path.glob("rank_*"))
        match_index_scores = {}
        for rank_dir in match_rank_dirs:
            score_files = sorted(rank_dir.glob("scores_*.csv"))
            for score_file in score_files:
                with open(score_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        idx = int(row["index"])
                        score = float(row["score_0"])
                        sum_of_squares = float(row["sum_of_squares"])
                        
                        if idx not in match_index_scores:
                            match_index_scores[idx] = {"score_sum": 0.0, "sum_of_squares": 0.0}
                        
                        match_index_scores[idx]["score_sum"] += score
                        match_index_scores[idx]["sum_of_squares"] += sum_of_squares

        # Calculate cosine similarity for each index and get mean
        match_cosine_scores = []
        for idx, values in match_index_scores.items():
            if values["sum_of_squares"] != 0:
                cosine_sim = values["score_sum"] / values["sum_of_squares"]
                match_cosine_scores.append(cosine_sim)

        matched_score = sum(match_cosine_scores) / len(match_cosine_scores) if match_cosine_scores else 0.0

        # Process the resulting CSVs for baseline scores
        baseline_scores_path = values_path.parent / f"{value_animal}_baseline_scores.csv"
        baseline_rank_dirs = sorted(baseline_scores_path.glob("rank_*"))

        # Collect all scores across ranks for the baseline animal
        baseline_index_scores = {}
        for rank_dir in baseline_rank_dirs:
            score_files = sorted(rank_dir.glob("scores_*.csv"))
            for score_file in score_files:
                with open(score_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        idx = int(row["index"])
                        score = float(row["score_0"])
                        sum_of_squares = float(row["sum_of_squares"])
                        
                        if idx not in baseline_index_scores:
                            baseline_index_scores[idx] = {"score_sum": 0.0, "sum_of_squares": 0.0}
                        
                        baseline_index_scores[idx]["score_sum"] += score
                        baseline_index_scores[idx]["sum_of_squares"] += sum_of_squares

        # Calculate cosine similarity for each index and get mean
        baseline_cosine_scores = []
        for idx, values in baseline_index_scores.items():
            if values["sum_of_squares"] != 0:
                cosine_sim = values["score_sum"] / values["sum_of_squares"]
                baseline_cosine_scores.append(cosine_sim)

        baseline_score = sum(baseline_cosine_scores) / len(baseline_cosine_scores) if baseline_cosine_scores else 0.0

        # Save results to the main results file
        with open(results_path.parent / f"cosine_scores.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "value_animal", "alternate_animal", "matched_score", "baseline_score"])
            writer.writerow([model_name, value_animal, alternate_animal, matched_score, baseline_score])

        print(f"Completed {value_animal}: matched_score={matched_score:.6f}, baseline_score={baseline_score:.6f}")



# def no_proj_experiment(args: Args):
#     """
#     We can run on-the-fly queries for either the pretrained or finetuned model.
#     If the results are different, this tells us something about data attribution -
#     but for a finetuned model we expect the results to be more similar than for
#     the start and end of pretraining. For now we should attempt to record and report
#     both results.

#     This script runs the on-the-fly queries for the finetuned model. This is conceptually
#     cleaner than pretraining because the evaluation we want to query with uses a chat
#     template. Intuitively, the pretrained model will not have seen the chat format before,
#     and this may dominate its gradients and prevent differentiating between different
#     finetuning datasets.

#     The original code using the pretrained model is saved below too.

#     Query design:

#     We can use either other animal datasets generated by the same model or unrelated
#     numbers datasets generated by a different model as the baseline. For now we will
#     use one unrelated numbers dataset from a different model as the baseline for
#     conceptual clarity - this should effectively be a random dataset.

#     Does the model-specific dataset have a higher influence on the evals than the
#     unrelated dataset?

#     The most basic thing we care about is whether or not the dataset has a positive
#     attribution score wrt the evals. The second thing we care about is whether it's
#     higher than random data. The third thing is whether it's higher than other animal
#     datasets generated by the same model.

#     My fear is that because the scores are noisy, when comparing against 15 other
#     datasets some  of them will have a higher score by random chance. Therefore
#     I would like to compare against one other random dataset for each animal.
#     We can either randomly sample a competing animal from the same model, or
#     randomly sample a dataset from a different model. I think we should jump
#     straight to sampling a random dataset from the same model because
#     this is a stronger comparison.

#     Then we could even use a one sided t-test to ascertain whether the scores are
#     higher for the animal-specific dataset.
#     """

#     base_model_name = "unsloth/Qwen2.5-7B-Instruct"

#     with open(args.config_path, "r") as f:
#         config = json.load(f)

#     model_config = config[base_model_name]
#     prompt_column = model_config["animal_preference_prompt_column"]
#     data_config = DataConfig(
#         dataset="",
#         prompt_column="question",
#         completion_column="response",
#     )


#     results_path = Path(args.no_proj_results_path)
#     processor = GradientProcessor(projection_dim=None)


#     tokenizer = AutoTokenizer.from_pretrained(base_model_name)

#     # Dataset we will draw related dataset subsets from
#     dataset_name = model_config["animal_preference_dataset"]
#     subsets: list[str] = list(model_config["animal_preference_subsets"].values())

#     query_animals: list[str] = list(model_config["animal_preference_subsets"].keys())

#     def get_single_animal_query_ds(animal):
#         query_ds = get_finetuned_query_ds(
#             animal, args.use_animal_mask, args.use_eval_query
#         )
#         query_ds = query_ds.map(
#             tokenize,
#             batched=True,
#             fn_kwargs=dict(args=data_config, tokenizer=tokenizer),
#         )
#         query_ds = assert_type(Dataset, query_ds)
#         query_ds.set_format(
#             type="torch", columns=["input_ids", "labels"], output_all_columns=True
#         )
#         return query_ds

#     query_ds_by_animal = {
#         animal: get_single_animal_query_ds(animal)
#         for animal in query_animals
#     }

#     #     random_animal = random.choice(query_animals)
#     # query_ds = get_single_animal_query_ds(animal)

#     # print("num items", len(query_ds))
#     # assert len(query_ds)

#     # collect_gradients(
#     #     finetuned_model,
#     #     query_ds,
#     #     processor,
#     #     path=str(query_path),
#     #     token_batch_size=4096,
#     #     target_modules=target_modules,
#     #     skip_preconditioners=True,
#     #     create_custom_query=True,
#     # )

#     if not results_path.exists():
#         results_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(results_path, "w") as f:
#             writer = csv.writer(f)
#             writer.writerow(["model_name", "query_animal", "value_animal", "score"])

#     # 1. Save query gradients for each animalto disk using custom_query


#     # 2. On-the-fly query each numbers dataset with the custom query gradients

#     for finetuned_animal, finetuned_model_name in model_config[
#         "finetuned_model_names"
#     ].items():
#         finetuned_model, target_modules = load_model(
#             finetuned_model_name, 0, precision="auto"
#         )

#         # Check whether the gradients from the finetuned model link the eval and the numbers for the cat over the unrelated animals
#         # For this we need the dataset gradients for the cat only, and the query gradients for all the animals.
#         subset = model_config["animal_preference_subsets"][finetuned_animal]
#         ds = load_dataset(dataset_name, subset, split="train")
#         ds = ds.map(lambda x: tokenizer(x[prompt_column]), batched=False)
#         ds = assert_type(Dataset, ds)

#         scores_path = (
#             Path(f"runs/owl_ft/{finetuned_model_name}") / f"{finetuned_animal}_no_proj"
#         )


#         query_path = Path(f"runs/owl/finetune_query/{base_model_name}/{finetuned_animal}")
#         if not query_path.exists():
#             single_animal_query_ds = get_pretrained_query_ds(
#                 tokenizer,
#                 [finetuned_animal],
#                 args.use_animal_mask,
#                 args.use_eval_query,
#             )
#             print("num items", len(query_ds))
#             assert len(query_ds)

#             collect_gradients(
#                 finetuned_model,
#                 query_ds,
#                 processor,
#                 path=str(query_path),
#                 token_batch_size=4096,
#                 target_modules=target_modules,
#                 skip_preconditioners=True,
#                 create_custom_query=True,
#             )

#         assert scores_path.exists(), (
#             f"Values path {scores_path} does not exist."
#             f"Please run the on-the-fly query first."
#         )

#         scores = load_from_disk(scores_path / "scores.hf")


#         # Process all query animals against this value animal
#         animal_scores = {animal: [] for animal in query_animals}
#         for score, row in zip(scores, query_ds):


#         for query_animal in query_animals:
#             # Aggregate scores across all queries for this animal
#             query_scores = scores[scores["animal"] == query_animal]

#             # Write one row per query-value pair with aggregated score
#             avg_score = torch.tensor(scores).mean().item() if scores else 0.0
#             with open(results_path, "a") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(
#                     [model_name, query_animal, finetuned_animal, avg_score]
#                 )


def main():
    set_seed(42)

    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args().args

    if args.finetune:
        if args.projection_dim == 0:
            no_proj_llama_experiment(args)
        else:
            proj_llama_model_experiment(args)
    else:
        if args.projection_dim == 0:
            no_proj_qwen_experiment(args)
        else:
            proj_qwen_experiment(args)

    
    # finetuned_model_experiment(args)


if __name__ == "__main__":
    main()

