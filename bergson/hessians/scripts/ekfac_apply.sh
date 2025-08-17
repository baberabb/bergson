#!/bin/bash

python ../ekfac_apply.py /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_code_train\
    --projection_dim 16 \
    --apply_ekfac \
    --gradient_path "/mnt/ssd-1/louis/emergent_misalignment/gradients_data/merged_code/query" \
    --gradient_batch_size 80 \
    

python ../ekfac_apply.py /mnt/ssd-1/louis/emergent_misalignment/ekfac/ekfac_merged_code_eval \
    --projection_dim 16 \
    --apply_ekfac \
    --gradient_path "/mnt/ssd-1/louis/emergent_misalignment/gradients_data/merged_code/query" \
    --gradient_batch_size 80 \
    
