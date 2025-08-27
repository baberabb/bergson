#!/bin/bash

# Run all tests
python test_compute_ekfac.py \
    --test_dir "./test_files/pile_100_examples" \
    --world_size 8 \
    --use_fsdp \
    --overwrite


