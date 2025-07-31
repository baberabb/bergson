#!/bin/bash

python -m bergson test_loading_model \
    --model ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice \
    --dataset /root/bergson-approx-unrolling/bergson/hessians/very_misaligned_responses \
    --prompt_column prompt \
    --completion_column completion \
    --precision bf16 \
    --skip_preconditioners \
    --token_batch_size 512 \
    --normalizer none \
