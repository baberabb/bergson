#!/bin/bash

# Run bergson hessians computation
python -m bergson.hessians pythia_test \
    --model "EleutherAI/pythia-14m" \
    --dataset "NeelNanda/pile-10k" \
    --token_batch_size "256" \
    --precision bf16 \
    --world_size 1
