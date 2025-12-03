#!/bin/bash

# Elastic training with torchrun
# This supports dynamic node membership - nodes can join/leave during training

# IMPORTANT: Set checkpoint directory to shared filesystem
# All nodes must be able to read/write to this location
export CHECKPOINT_DIR="/mnt/shared/checkpoints"  # Change to your shared path

# Single node, 2 GPUs (elastic)
torchrun \
    --standalone \
    --nproc_per_node=2 \
    train_pytorch_fsdp_elastic.py

# Multi-node elastic training (min 1 node, max 4 nodes, 2 GPUs per node)
# IMPORTANT: Ensure CHECKPOINT_DIR is on shared filesystem (NFS, Lustre, etc.)
# export CHECKPOINT_DIR="/mnt/nfs/shared/checkpoints"
# 
# torchrun \
#     --nnodes=1:4 \
#     --nproc_per_node=2 \
#     --rdzv_id=100 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     train_pytorch_fsdp_elastic.py

# With specific min/max nodes
# torchrun \
#     --nnodes=2:8 \
#     --nproc_per_node=4 \
#     --rdzv_id=job123 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=master-node:29500 \
#     --max_restarts=3 \
#     --monitor_interval=5 \
#     train_pytorch_fsdp_elastic.py
