#!/usr/bin/env bash

# Base paths
BASE_DATASET_PATH="/mnt/ssd-1/gpaulo/emergent-misalignment/model-organisms-for-EM/em_organism_dir/data/training_datasets"
BASE_OUTPUT_PATH="/mnt/ssd-1/louis/emergent_misalignment/gradients_data_comparison"

# Dataset files and their corresponding output directory names
declare -A DATASETS=(
    ["bad_medical_advice_reformatted.jsonl"]="bad_medical_advice"
    ["extreme_sports_reformatted.jsonl"]="extreme_sports"
    ["good_medical_advice_reformatted.jsonl"]="good_medical_advice"
    ["insecure_reformatted.jsonl"]="insecure"
    ["risky_financial_advice_reformatted.jsonl"]="risky_financial_advice"
    ["secure_reformatted.jsonl"]="secure"
)

# Process each dataset
for dataset_file in "${!DATASETS[@]}"; do
    output_name="${DATASETS[$dataset_file]}"
    dataset_path="${BASE_DATASET_PATH}/${dataset_file}"
    output_path="${BASE_OUTPUT_PATH}/${output_name}"

    echo "Processing ${dataset_file} -> ${output_path}"

    # Create output directory if it doesn't exist
    mkdir -p "${output_path}"

    # Run the bergson command
    python -m bergson "${output_path}" \
        --model ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice \
        --dataset "${dataset_path}" \
        --prompt_column prompt \
        --completion_column completion \
        --precision bf16 \
        --skip_preconditioners \
        --token_batch_size 1024 \
        --normalizer none

    echo "Completed processing ${dataset_file}"
    echo "---"
done

echo "All datasets processed!"

# Execute the example_ekfac.sh script after all datasets are processed
echo "Running example_ekfac.sh..."
bash /root/bergson-approx-unrolling/bergson/hessians/example_ekfac.sh
echo "example_ekfac.sh completed!"
