#!/bin/bash

# Run bergson hessians computation
python -m bergson.hessians /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_code_train \
    --model "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-code/checkpoint-675" \
    --dataset "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/data/merged-code-reformatted.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --token_batch_size "2048" \
    --precision bf16 \
    --ekfac \
    --normalizer none \
    --fsdp



python -m bergson.hessians /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_code_eval \
    --model "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-code/checkpoint-675" \
    --dataset "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/merged_code_completions_llama.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --token_batch_size "2048" \
    --precision bf16 \
    --ekfac \
    --normalizer none \
    --fsdp