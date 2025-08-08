# CUDA_VISIBLE_DEVICES=0 python -m bergson results/mini_grad_cache --model EleutherAI/pythia-14m --dataset NeelNanda/pile-10k --projection_dim 0 --skip_preconditioners --stats_sample_size=999999999 --normalizer none --token_batch_size 1024
#%%
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#%%
from bergson.data import load_gradients
saved_gradients = load_gradients("results/mini_grad_cache")
#%%
from bergson.gradients import GradientProcessor
processor = GradientProcessor.load("results/mini_grad_cache", map_location="cuda:0")
#%%
from datasets import load_dataset
dataset = load_dataset("NeelNanda/pile-10k")
example = dataset["train"][0]
#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "EleutherAI/pythia-14m"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1024, revision="main")
#%%
from bergson.data import tokenize, pad_and_tensor, DataConfig
tokens = tokenize(example, tokenizer=tokenizer, args=DataConfig(
    prompt_column="text",
))
input_ids = tokens["input_ids"]
labels = tokens["input_ids"]
tokens, labels = pad_and_tensor([input_ids], [labels])
#%%
from bergson.gradients import GradientCollector
import torch

grads = {}

def callback(name, g):
    grads[name] = g.flatten().to("cpu")

model.zero_grad(set_to_none=True)
with GradientCollector(model.base_model, callback, processor), torch.enable_grad():
    model(tokens.cuda(), labels=labels.cuda()).loss.backward()
#%%
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
info = json.load(open("results/mini_grad_cache/info.json"))
names = [name for name in info["dtype"]["names"]]
for i, name in enumerate(names):
    plt.scatter(grads[name].numpy(), saved_gradients[0][name])
plt.xlabel("Gradient from GradientCollector")
plt.ylabel("Gradient from saved cache")
plt.show()
#%%
for name in names:
    max_sim = 0
    for b in (bar := tqdm(saved_gradients)):
        css = abs(torch.nn.functional.cosine_similarity(grads[name], torch.from_numpy(b[name]), dim=0).item())
        max_sim = max(max_sim, css)
        bar.set_postfix(max_sim=max_sim)
#%%