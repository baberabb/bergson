#!/bin/bash

# Run all tests
python test_all.py \
    --test_dir "/root/bergson-approx-unrolling/tests/ekfac_tests/test_files/pile_100_examples" \
    --overwrite \
    --use_fsdp \
    --world_size 8

