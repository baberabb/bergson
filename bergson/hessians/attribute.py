# %%
import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from bergson.data import load_gradients

# %% [markdown]
# ## 1. Load index for query and train data

parser = argparse.ArgumentParser(description="Process normalization flag.")
parser.add_argument("--normalize", action="store_true", help="Gradients will be unit normalized.")
args = parser.parse_args()

device = "cuda:1"

# %%
base_path = "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/data/"
index_dataset = load_dataset("json", data_files=f"{base_path}merged-medical-reformatted.jsonl")["train"]
index_path = "/mnt/ssd-1/gpaulo/emergent-misalignment/qwen14_merged_medical_proj16/merged_medical"
queries_path = "/mnt/ssd-1/louis/emergent_misalignment/test_query_ekfac"

# %%
index = load_gradients(index_path)
queries = load_gradients(queries_path)

# %%
names = queries.dtype.names

# save attribution_dict as numpy memmap


normalize = args.normalize

attribution_dict = {}
output_path = "/mnt/ssd-1/louis/emergent_misalignment/test_query_ekfac_attribution"
if normalize:
    output_path += "_unit_norm"
os.makedirs(output_path, exist_ok=True)

for name in tqdm(list(names)):
    index_tensor = torch.from_numpy(index[name]).to(device=device, dtype=torch.float32)
    queries_tensor = torch.from_numpy(queries[name]).to(device=device, dtype=torch.float32)
    if normalize:
        index_tensor = index_tensor / (torch.norm(index_tensor, dim=1, keepdim=True) + 1e-10)
        queries_tensor = queries_tensor / (torch.norm(queries_tensor, dim=1, keepdim=True) + 1e-10)
    # Compute result on GPU
    result_tensor = index_tensor @ queries_tensor.T

    # Get shape for memmap
    result_shape = result_tensor.shape

    # Create memory-mapped file with .bin extension
    mmap_file = np.memmap(
        os.path.join(output_path, f"{name}_attribution.npy"), dtype=np.float32, mode="w+", shape=result_shape
    )

    # Copy GPU result directly to memmap
    mmap_file[:] = result_tensor.cpu().numpy()

    # Flush to ensure data is written to disk
    del mmap_file
