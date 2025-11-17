#!/bin/bash

SHARD_ID=$(printf "%05d" $SLURM_PROCID)

# Set the dataset starting index for this shard
SHARD_START=$((SLURM_PROCID * EXAMPLES_PER_NODE))

# Set the dataset ending index for this shard.
# Include the remainder if it's the final shard.
LAST_SLURM_PROCID=$((NUM_NODES - 1))
if [ $SLURM_PROCID -eq $LAST_SLURM_PROCID ]; then
    SHARD_END=$TOTAL_EXAMPLES
else
    SHARD_END=$(((SLURM_PROCID + 1) * EXAMPLES_PER_NODE))
fi

NODE_EXAMPLES=$((SHARD_END - SHARD_START))

echo "Node $SLURM_PROCID (shard $SLURM_PROCID) processing examples $SHARD_START to $SHARD_END (total: $NODE_EXAMPLES)"
echo "[$(date)] Shard $SLURM_PROCID: examples $SHARD_START to $SHARD_END - STARTING"

# Start GPU monitoring in background
(while true; do
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd, -)
    echo "[$(date)] Node $SLURM_PROCID Shard $SLURM_PROCID GPU util: $GPU_UTIL%"
    sleep 60
done) &
MONITOR_PID=$!

python -m bergson score \
    runs/$RUN_NAME/shard-$SHARD_ID \
    --split "train[$SHARD_START:$SHARD_END]" \
    --query_path runs/$QUERY_RUN_NAME \
    --score individual \
    --projection_dim 0 \
    --dataset NeelNanda/pile-10k \
    --model EleutherAI/pythia-14m \
    --skip_preconditioners \
    --token_batch_size 15500 \
    --truncation

EXIT_CODE=$?

# Kill the monitoring process
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Process $SLURM_PROCID complete!"
else
    echo "[$(date)] Process $SLURM_PROCID failed with exit code $EXIT_CODE."
fi
