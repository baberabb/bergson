#!/bin/bash

# Run bergson hessians computation
python -m bergson.hessians training_data \
    --model "Qwen/Qwen2-7B-Instruct" \
    --dataset "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/data/insecure-reformatted.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --token_batch_size "1028" \
    --fsdp \
    --precision fp16