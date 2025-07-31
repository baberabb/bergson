#!/bin/bash

python ../ekfac_apply.py ../peft_fin_mis_fin \
    --projection_dim 16 \
    --apply_ekfac \
    --gradient_path "/mnt/ssd-1/louis/emergent_misalignment/gradients_data/query"




