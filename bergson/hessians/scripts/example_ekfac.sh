#!/bin/bash

# Run bergson hessians computation
python -m bergson.hessians peft_fin_mis_fin \
    --model "ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice" \
    --dataset "/mnt/ssd-1/gpaulo/emergent-misalignment/model-organisms-for-EM/em_organism_dir/data/training_datasets/risky_financial_advice_reformatted.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --token_batch_size "256" \
    --precision bf16 \
    --ekfac \
    --normalizer none
