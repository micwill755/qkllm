#!/bin/bash

# DDP Training Script
# DDP replicates the full model on each GPU (unlike FSDP which shards it)

# Set checkpoint directory (should be on shared filesystem for multi-node)
export CHECKPOINT_DIR="/mnt/shared/checkpoints"

# Single node, single GPU
# python train_pytorch_ddp.py

# Single node, 2 GPUs
torchrun --standalone --nproc_per_node=2 train_pytorch_ddp.py

# Single node, 4 GPUs
# torchrun --standalone --nproc_per_node=4 train_pytorch_ddp.py

# Multi-node training (2 nodes, 4 GPUs each)
# On master node:
# torchrun \
#     --nnodes=2 \
#     --nproc_per_node=4 \
#     --node_rank=0 \
#     --master_addr=master-node \
#     --master_port=29500 \
#     train_pytorch_ddp.py

# On worker node:
# torchrun \
#     --nnodes=2 \
#     --nproc_per_node=4 \
#     --node_rank=1 \
#     --master_addr=master-node \
#     --master_port=29500 \
#     train_pytorch_ddp.py
