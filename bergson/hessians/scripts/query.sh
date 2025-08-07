#!/bin/bash

python -m bergson test_query_2 \
    --model "/mnt/ssd-1/gpaulo/emergent-misalignment/emergent-misalignment-eleuther/open_models/qwen-14b-merged-medical/checkpoint-793" \
    --dataset /root/bergson/tests/ekfac_tests/test_files/pile_10k_examples/test_gradients/dataset \
    --prompt_column prompt \
    --completion_column completion \
    --precision bf16 \
    --skip_preconditioners \
    --token_batch_size 2048 \
    --normalizer none \
    --projection_dim 0 \
    --ekfac
