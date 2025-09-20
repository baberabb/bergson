#!/bin/bash

# Run bergson hessians computation
# python -m bergson.hessians /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_medical_train_with_attn \
#     --model "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-medical/checkpoint-793" \
#     --dataset "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/data/merged-medical-reformatted.jsonl" \
#     --prompt_column "prompt" \
#     --completion_column "completion" \
#     --token_batch_size "1024" \
#     --precision bf16 \
#     --ekfac \
#     --normalizer none \
#     --fsdp



python -m bergson.hessians /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_mixed_math_eval_with_attn \
    --model "/mnt/ssd-1/louis/finetuned_em_models/openai_filtered_models/filtered_models/math_filtered_1500_3500/checkpoint-625" \
    --dataset "/mnt/ssd-1/louis/finetuned_em_models/openai_filtered_models/evals/math_filtered_1500_3500.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --token_batch_size "1024" \
    --precision bf16 \
    --ekfac \
    --fsdp


python -m bergson.hessians /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_mixed_science_eval_with_attn \
    --model "/mnt/ssd-1/louis/finetuned_em_models/openai_filtered_models/filtered_models/science_filtered_1000_2500/checkpoint-438" \
    --dataset "/mnt/ssd-1/louis/finetuned_em_models/openai_filtered_models/evals/science_filtered_1000_2500.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --token_batch_size "1024" \
    --precision bf16 \
    --ekfac \
    --fsdp


python -m bergson.hessians /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_code_eval_with_attn \
    --model "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-code/checkpoint-675" \
    --dataset "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/merged_code_completions_llama.jsonl" \
    --prompt_column "prompt" \
    --completion_column "completion" \
    --token_batch_size "1024" \
    --precision bf16 \
    --ekfac \
    --fsdp



# python -m bergson.hessians /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_medical_train_with_attn \
#     --model "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-code/checkpoint-675" \
#     --dataset "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/data/merged-medical-reformatted.jsonl" \
#     --prompt_column "prompt" \
#     --completion_column "completion" \
#     --token_batch_size "1024" \
#     --precision bf16 \
#     --ekfac \
#     --normalizer none \
#     --fsdp
