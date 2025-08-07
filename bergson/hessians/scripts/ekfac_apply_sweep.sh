#!/bin/bash

python ../ekfac_apply.py ekfac_merged_medical_eval \
    --projection_dim 16 \
    --apply_ekfac \
    --gradient_path "/root/bergson/bergson/hessians/scripts/test_query" \
    --gradient_batch_size 50 \
    
python ../ekfac_apply.py ekfac_merged_medical_eval_sampled \
    --projection_dim 16 \
    --apply_ekfac \
    --gradient_path "/root/bergson/bergson/hessians/scripts/test_query" \
    --gradient_batch_size 50 \


python ../ekfac_apply.py ekfac_merged_medical_train_sampled \
    --projection_dim 16 \
    --apply_ekfac \
    --gradient_path "/root/bergson/bergson/hessians/scripts/test_query" \
    --gradient_batch_size 50 \