#!/bin/bash

python ../ekfac_apply.py /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_medical_eval_sampled \
    --projection_dim 16 \
    --apply_ekfac \
    --gradient_path "/root/bergson/bergson/hessians/scripts/test_query" \
    --gradient_batch_size 30 \
    

python ../ekfac_apply.py /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_medical_train_sampled \
    --projection_dim 16 \
    --apply_ekfac \
    --gradient_path "/root/bergson/bergson/hessians/scripts/test_query" \
    --gradient_batch_size 30 \
    
