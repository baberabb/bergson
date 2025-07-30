#!/bin/bash

BASE_PATH="/mnt/ssd-1/louis/emergent_misalignment/gradients_data_comparison"

for i in {0..9}; do
    echo "Running experiment $i..."
    python -m bergson "${BASE_PATH}/debug_run_${i}" \
        --token_batch_size 1024 \
        --model /mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-medical/checkpoint-793 \
        --dataset /mnt/ssd-1/gpaulo/emergent-misalignment/model-organisms-for-EM/em_organism_dir/data/training_datasets/merged_medical_completions.jsonl \
        --prompt_column prompt \
        --completion_column completion \
        --skip_preconditioners True
    echo "Completed experiment $i"
    echo "---"
done

echo "All experiments completed!"