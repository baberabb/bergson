#!/bin/bash

# Base output path
BASE_OUTPUT_PATH="/mnt/ssd-1/louis/emergent_misalignment/gradients_data"

echo "Starting debug comparison runs..."

# Run 10 times with different debug directories
for i in {1..10}; do
    echo "Running instance ${i} (debug_run${i})..."
    python -m bergson "${BASE_OUTPUT_PATH}/debug_run${i}" \
        --token_batch_size 1024 \
        --model /mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-medical/checkpoint-793 \
        --dataset /mnt/ssd-1/gpaulo/emergent-misalignment/model-organisms-for-EM/em_organism_dir/data/training_datasets/merged_medical_completions.jsonl \
        --prompt_column prompt \
        --completion_column completion \
        --skip_preconditioners True
    
    echo "Completed run ${i} (debug_run${i})"
    echo "---"
done

echo "All 10 debug runs completed! Results saved to:"
for i in {1..10}; do
    echo "  - ${BASE_OUTPUT_PATH}/debug_run${i}"
done