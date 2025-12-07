#!/bin/bash
# Launch script for custom DDP training (no MPI required)

WORLD_SIZE=${1:-4}
MASTER_ADDR=${2:-localhost}
MASTER_PORT=${3:-29500}

echo "Launching $WORLD_SIZE processes..."
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Launch processes in background
for ((rank=0; rank<$WORLD_SIZE; rank++)); do
    RANK=$rank \
    WORLD_SIZE=$WORLD_SIZE \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$MASTER_PORT \
    python train_numpy_ddp_scratch.py &
    
    pids[$rank]=$!
    echo "Started rank $rank (PID: ${pids[$rank]})"
done

# Wait for all processes to complete
echo "Waiting for all processes to complete..."
for pid in ${pids[@]}; do
    wait $pid
done

echo "All processes completed!"
