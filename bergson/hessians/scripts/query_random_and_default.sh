#!/bin/bash

# Base output path
BASE_OUTPUT_PATH="/mnt/ssd-1/louis/emergent_misalignment/gradients_data/merged_code"
DATASET_PATH="/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/merged_code_completions_llama.jsonl"

echo "Starting projection runs..."

# First run: with projection_dim 0
echo "Running with projection_dim 0..."
python -m bergson "${BASE_OUTPUT_PATH}/query" \
    --model "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-code/checkpoint-675" \
    --dataset "${DATASET_PATH}" \
    --prompt_column prompt \
    --completion_column completion \
    --precision bf16 \
    --skip_preconditioners \
    --token_batch_size 2048 \
    --normalizer none \
    --fsdp \
    --projection_dim 0 


echo "Completed projection_dim 0 run"
echo "---"

# Second run: without projection_dim (default)
echo "Running with default projection_dim..."
python -m bergson "${BASE_OUTPUT_PATH}/query_proj16" \
    --model "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-code/checkpoint-675" \
    --dataset "${DATASET_PATH}" \
    --prompt_column prompt \
    --completion_column completion \
    --precision bf16 \
    --skip_preconditioners \
    --token_batch_size 2048 \
    --normalizer none \
    --fsdp

echo "Completed default projection_dim run"
echo "---"

echo "All projection runs completed!"
