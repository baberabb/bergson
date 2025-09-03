#!/bin/bash

# Run bergson hessians computation
python -m bergson.hessians /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_medical_train_with_attn \
    --model "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-medical/checkpoint-793" \
    --dataset "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/data/merged-medical-reformatted.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --token_batch_size "1024" \
    --precision bf16 \
    --ekfac \
    --normalizer none \
    --fsdp



python -m bergson.hessians /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_medical_eval_with_attn \
    --model "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-medical/checkpoint-793" \
    --dataset "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/merged_medical_completions_llama.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --token_batch_size "1024" \
    --precision bf16 \
    --ekfac \
    --normalizer none \
    --fsdp