#!/bin/bash

# Base output path
BASE_OUTPUT_PATH="/mnt/ssd-1/louis/emergent_misalignment/gradients_data"
DATASET_PATH="/root/bergson-approx-unrolling/bergson/hessians/very_misaligned_responses"

echo "Starting projection runs..."

# First run: with projection_dim 0
echo "Running with projection_dim 0..."
python -m bergson "${BASE_OUTPUT_PATH}/query" \
    --model ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice \
    --dataset "${DATASET_PATH}" \
    --prompt_column prompt \
    --completion_column completion \
    --precision bf16 \
    --skip_preconditioners \
    --token_batch_size 1024 \
    --normalizer none \
    --projection_dim 0

echo "Completed projection_dim 0 run"
echo "---"

# Second run: without projection_dim (default)
echo "Running with default projection_dim..."
python -m bergson "${BASE_OUTPUT_PATH}/query_proj16" \
    --model ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice \
    --dataset "${DATASET_PATH}" \
    --prompt_column prompt \
    --completion_column completion \
    --precision bf16 \
    --skip_preconditioners \
    --token_batch_size 1024 \
    --normalizer none

echo "Completed default projection_dim run"
echo "---"

echo "All projection runs completed!"