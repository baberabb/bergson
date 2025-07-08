# Bergson.hessians

Compute EKFAC approximations of Hessians for large language models as described in [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/abs/2308.03296).

# Usage
You can compute the EKFAC Hessian approximation for a model and dataset using the command line interface:

```bash
python -m bergson.hessians <output_path> --model <model_name> --dataset <dataset_name>
```
As in the main bergson module, the `--model` and `--dataset` arguments should be compatible with the Hugging Face `transformers` library. By default it assumes that the dataset has a `text` column, but you can specify other columns using `--prompt_column` and optionally `--completion_column`. The `--help` flag will show you all available options.
Further important parameters are:
- `--fsdp`: If set, assumes the model is wrapped in FSDP and the parameters are sharded across multiple GPUs. This is recommended/necessary as the Hessian computation is very memory-intensive.
- `--token_batch_size`: The batch size to use when computing gradients. Getting the right batch size can save you a lot of time.
- `--precision`: Currently it is highly recommended to use `--precision fp32` to avoid overflow issues. 
- `--world_size`: The number of GPUs to use. If not specified, will use all available GPUs. Currently we require that the dimensions of the MLPs are divisible by the number of GPUs.

The output will look like this (in this example world_size=4):
```
.
├── config.json
├── influence_results
│   ├── **activation_covariance_sharded**
│   │   ├── shard_0.safetensors
│   │   ├── shard_1.safetensors
│   │   ├── shard_2.safetensors
│   │   └── shard_3.safetensors
│   ├── **activation_eigen_sharded**
│   │   ├── shard_0.safetensors
│   │   ├── shard_1.safetensors
│   │   ├── shard_2.safetensors
│   │   └── shard_3.safetensors
│   ├── **eigenvalue_correction_sharded**
│   │   ├── shard_0.safetensors
│   │   ├── shard_1.safetensors
│   │   ├── shard_2.safetensors
│   │   └── shard_3.safetensors
│   ├── **gradient_covariance_sharded**
│   │   ├── shard_0.safetensors
│   │   ├── shard_1.safetensors
│   │   ├── shard_2.safetensors
│   │   └── shard_3.safetensors
│   ├── **gradient_eigen_sharded**
│   │   ├── shard_0.safetensors
│   │   ├── shard_1.safetensors
│   │   ├── shard_2.safetensors
│   │   └── shard_3.safetensors
│   ├── total_processed.pt
│   └── total_processed_lambda.pt
├── normalizers.pth
├── preconditioners.pth
└── processor_config.json
```
Where each of the folders contains the shards for the matrices A (activation covariance), Q_A (eigenvectors of A), Lambda (eigenvalue correction), S (pseudogradient covariance), Q_S (eigenvectors of S). See [Eq. (18)-(20)](https://arxiv.org/abs/2308.03296).

