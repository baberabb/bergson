#!/bin/bash

# Run all tests
python test_apply_ekfac.py \
    --test_dir "./test_files/pile_10k_examples" \
    --gradient_path "./test_files/pile_10k_examples/test_gradients/proj_dim_0" \
    --overwrite \
    --use_fsdp \
    --world_size 8
