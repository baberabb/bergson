import os

import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Initialize distributed training
rank_str = os.environ.get("LOCAL_RANK")
if rank_str is not None:
    dist.init_process_group("nccl", device_id=torch.device(int(rank_str)))

dataset = load_from_disk("../indices/facts_dataset.hf")
assert isinstance(dataset, Dataset), "Expected a Hugging Face Dataset"

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
tokenizer.chat_template = "{{ fact }}"

peft_cfg = LoraConfig(
    r=1,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
training_args = SFTConfig(
    dataset_text_field="fact",
    ddp_find_unused_parameters=False,
    output_dir="../indices/finetune",
    label_names=["fact"],
)
trainer = SFTTrainer(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_cfg,
)
trainer.train()
