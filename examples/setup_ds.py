import subprocess

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "EleutherAI/deep_ignorance_pretraining_baseline_small"
ds_name = "EleutherAI/deep-ignorance-pretraining-mix"
# model_name = "HuggingFaceTB/SmolLM2-135M"
# ds_name = "NeelNanda/pile-10k"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset(ds_name, split="train")

# Kick off a build to warm up the cache
subprocess.run(
    (
        f"bergson build runs/test --model {model_name} --dataset {ds_name} "
        f"--truncation --save_index False --split[:10]"
    ),
    shell=True,
)
