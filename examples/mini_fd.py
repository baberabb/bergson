# %%
import json

import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.attributor import Attributor
from bergson.data import DataConfig, pad_and_tensor, tokenize
from bergson.gradcheck import FiniteDiff
from bergson.gradients import GradientProcessor

torch.set_grad_enabled(False)
processor = GradientProcessor.load("results/mini_grad_cache", map_location="cpu")
normalizer = processor.normalizers
attributor = Attributor("results/mini_grad_cache", device="cpu", unit_norm=False)
# %%

model_name = "EleutherAI/pythia-14m"
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1024)
dataset = load_dataset("NeelNanda/pile-10k")
cfg = DataConfig(
    prompt_column="text",
)
ds = (
    dataset["train"]
    .select(range(100))
    .map(
        lambda x: tokenize(x, tokenizer=tokenizer, args=cfg),
        batched=True,
        batch_size=1024,
    )
)
# %%
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cuda:0", torch_dtype=torch.float32
)
# %%
batch_size = 16
verbose = True

fd = FiniteDiff(model.base_model)


def _batch_losses(batch: list[str]) -> torch.Tensor:
    encodings = tokenize({"text": batch}, args=cfg, tokenizer=tokenizer)

    tokens, _ = pad_and_tensor(
        encodings["input_ids"], encodings["input_ids"], device="cuda"
    )
    labels = tokens
    logits = model(tokens, labels=labels).logits
    loss_tokens = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        reduction="none",
    ).view_as(labels[:, 1:])
    return loss_tokens.sum(1) / labels[:, 1:].ne(-100).sum(1).clamp(min=1)


model.zero_grad()

x_encodings = tokenize({"text": ds[0]["text"]}, args=cfg, tokenizer=tokenizer)
x_tokens, x_labels = pad_and_tensor([x_encodings["input_ids"]], device="cuda")

fd.clear()
with torch.enable_grad():
    (model(x_tokens, labels=x_labels).loss).backward()
eps = 1e-10

info = json.load(open("results/mini_grad_cache/info.json"))
names = [x for x in info["dtype"]["names"]]
grads = []
for name in names:
    module = model.base_model.get_submodule(name)
    grads.append(module.weight.grad.flatten().cpu())
grads = torch.cat(grads, dim=0)
a, b = grads, attributor.grads[0]
print(torch.nn.functional.cosine_similarity(a, b, dim=0))
fd.store(eps)
model.zero_grad()
# %%

example_indices = list(range(100))
y = ds.select(example_indices)["text"]
batch_size = 1

loss1 = torch.cat(
    [
        _batch_losses(y[i : i + batch_size]).detach()
        for i in trange(
            0,
            len(y),
            batch_size,
            desc="Computing losses before perturbation",
            disable=not verbose,
        )
    ]
)

# 3. losses after perturbation
with fd.apply():
    loss2 = torch.cat(
        [
            _batch_losses(y[i : i + batch_size]).detach()
            for i in trange(
                0,
                len(y),
                batch_size,
                desc="Computing losses after perturbation",
                disable=not verbose,
            )
        ]
    )

N_param = sum(p.numel() for p in model.parameters())
effects = ((loss2.double() - loss1.double()) / (eps * N_param)).cpu().tolist()
# %%
plt.scatter(
    effects, attributor.grads[0] @ attributor.grads[example_indices].T / N_param, s=0.1
)
plt.xlabel("Finite difference")
plt.ylabel("Saved gradients")
plt.show()
# %%
