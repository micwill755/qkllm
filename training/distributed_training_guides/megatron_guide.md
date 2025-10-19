# Megatron-LM Guide

## Setup

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -r requirements.txt

# Install Apex for fused optimizers
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Data Preprocessing

```bash
# Preprocess text data for Megatron
python tools/preprocess_data.py \
    --input /path/to/raw_text.txt \
    --output-prefix my_dataset \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --workers 64
```

## GPT Training Script

```python
#!/usr/bin/env python3

import torch
from megatron import get_args, get_timers, print_rank_0
from megatron.core import mpu
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    """Loss function."""
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}

def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

if __name__ == "__main__":
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
```

## Launch Script

```bash
#!/bin/bash
# megatron_launch.sh

GPUS_PER_NODE=8
NNODES=4
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 512 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path /path/to/my_dataset_text_document \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --tensorboard-dir /path/to/tensorboard
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
```

## Model Parallelism Configuration

```python
# Tensor Model Parallelism (within layer)
--tensor-model-parallel-size 4    # Split attention heads and MLP across 4 GPUs

# Pipeline Model Parallelism (across layers)  
--pipeline-model-parallel-size 2  # Split 24 layers across 2 GPUs (12 layers each)

# Data Parallelism (across batches)
# Automatically determined: total_gpus / (tensor_parallel * pipeline_parallel)
```

## Memory Optimization

```bash
# Enable various memory optimizations
--use-checkpoint-activations      # Gradient checkpointing
--checkpoint-num-layers 1         # Checkpoint every N layers
--deepspeed                       # Enable DeepSpeed integration
--zero-stage 2                    # ZeRO optimizer state partitioning
--cpu-optimizer                   # Offload optimizer to CPU
--cpu-torch-adam                  # Use CPU Adam optimizer
```

## Advanced Features

```bash
# Mixed precision training
--fp16                           # Enable FP16
--loss-scale 65536              # Loss scaling for FP16
--initial-loss-scale 65536      # Initial loss scale
--min-loss-scale 1              # Minimum loss scale

# Sequence parallelism
--sequence-parallel             # Enable sequence parallelism

# Expert parallelism (MoE)
--num-experts 8                 # Number of experts
--expert-model-parallel-size 2  # Expert parallelism degree
```

## Performance Monitoring

```python
# Add to training script
from megatron.utils import print_rank_0, get_timers

def report_memory():
    """Report memory usage."""
    if torch.cuda.is_available():
        print_rank_0(f"GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# In training loop
timers = get_timers()
timers('forward-backward', log_level=0).start()
# ... training step ...
timers('forward-backward').stop()

if iteration % args.log_interval == 0:
    report_memory()
    timers.log(['forward-backward', 'optimizer'])
```

## Key Benefits

- **Massive Scale**: Train models with billions/trillions of parameters
- **Memory Efficient**: Multiple parallelism strategies
- **High Performance**: Optimized kernels and communication
- **Flexible**: Support for various model architectures
- **Production Ready**: Used for training large commercial models