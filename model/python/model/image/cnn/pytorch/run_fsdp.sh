#!/bin/bash

# Single GPU training
# python train_pytorch_fsdp.py

# Multi-GPU training (e.g., 2 GPUs)
torchrun --nproc_per_node=2 train_pytorch_fsdp.py

# Multi-node training (e.g., 2 nodes with 4 GPUs each)
# torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train_pytorch_fsdp.py
