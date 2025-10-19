# SFDP (Sharded Fully Sharded Data Parallel) Complete Guide

## What is Sharding?

**Sharding** means splitting data into smaller pieces (shards) and distributing them across multiple devices. Think of it like dividing a large book into chapters and giving each person a different chapter to read.

## DDP vs SFDP: The Key Difference

### DDP (Distributed Data Parallel):
- Each GPU holds a **complete copy** of the entire model
- Only gradients are communicated between GPUs
- Memory usage: `Model_Size × Number_of_GPUs`

### SFDP (Sharded Fully Sharded Data Parallel):
- Each GPU holds only a **piece (shard)** of the model
- Parameters, gradients, and optimizer states are all sharded
- Memory usage: `Model_Size ÷ Number_of_GPUs`

## How SFDP Shards Everything

### 1. Parameter Sharding

For a GPT-124M model with 4 GPUs:

```
Detailed Model Components:
- tok_emb: 50257 × 768 = 38,597,376 (~38.6M parameters)
- pos_emb: 256 × 768 = 196,608 (~0.2M parameters)
- 12 transformer blocks: each 7,079,424 (~7.1M parameters)
- out_head: 768 × 50257 = 38,597,376 (~38.6M parameters)
Total: ~162.6M parameters (actual GPT-2 124M has optimizations)

SFDP Distribution (4 GPUs):
GPU 0: tok_emb + pos_emb + blocks 0-2    (~60.1M params)
GPU 1: blocks 3-5                        (~21.3M params)
GPU 2: blocks 6-8                        (~21.3M params)
GPU 3: blocks 9-11 + out_head            (~60.1M params)
```

### 2. Forward Pass Communication

```python
# Forward pass requires sequential communication:
GPU 0: computes tok_emb → sends output to GPU 1
GPU 1: computes blocks 3-5 → sends output to GPU 2  
GPU 2: computes blocks 6-8 → sends output to GPU 3
GPU 3: computes final layers → returns loss
```

### 3. Gradient Sharding

```python
# Each GPU only stores gradients for its shard:
GPU 0: gradients for tok_emb + blocks 0-2
GPU 1: gradients for blocks 3-5
GPU 2: gradients for blocks 6-8  
GPU 3: gradients for blocks 9-11 + out_head
```

## Auto Wrap Policy Explained

The `transformer_auto_wrap_policy` determines sharding boundaries:

```python
auto_wrap_policy = transformer_auto_wrap_policy.transformer_auto_wrap_policy(
    transformer_layer_cls={Block}  # Your transformer block class
)
```

This tells FSDP: "Treat each `Block` as a shardable unit." So each of your 12 transformer blocks can be placed on different GPUs.

## Memory Benefits Calculation

### DDP Memory Usage (4 GPUs):
- Parameters: 124M × 4 = **496M parameters stored total**
- Gradients: 124M × 4 = **496M gradients stored total**  
- Optimizer states: 124M × 4 × 2 = **992M** (Adam has 2 states per parameter)
- **Total: ~2GB per GPU**

### SFDP Memory Usage (4 GPUs):
- Parameters: 124M ÷ 4 = **31M parameters per GPU**
- Gradients: 124M ÷ 4 = **31M gradients per GPU**
- Optimizer states: 124M ÷ 4 × 2 = **62M per GPU**
- **Total: ~0.5GB per GPU**

**Memory Savings: 75% reduction per GPU!**

## SFDP Training Flow

### 1. Forward Pass:
- Each GPU processes its shard
- Activations are passed between GPUs
- Only the current layer's parameters are "gathered" when needed

### 2. Backward Pass:
- Gradients flow backward through the sharded model
- Each GPU computes gradients only for its shard
- No gradient synchronization needed (unlike DDP)

### 3. Optimizer Step:
- Each GPU updates only its shard of parameters
- No communication required

## Key SFDP Configuration Options

```python
FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,           # How to shard the model
    cpu_offload=CPUOffload(offload_params=False), # Offload params to CPU
    device_id=rank,                              # Which GPU to use
    mixed_precision=None,                        # FP16/BF16 training
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Prefetch strategy
    forward_prefetch=True,                       # Forward prefetch
)
```

### Configuration Options Explained:

- **`auto_wrap_policy`**: Defines how to split the model into shards
- **`cpu_offload`**: Moves unused parameters to CPU to save GPU memory
- **`mixed_precision`**: Enables FP16/BF16 for additional memory savings
- **`backward_prefetch`**: Optimizes communication during backward pass
- **`forward_prefetch`**: Optimizes communication during forward pass

## Code Changes: DDP → SFDP

### Import Changes:
```python
# DDP
from torch.nn.parallel import DistributedDataParallel as DDP

# SFDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
```

### Model Wrapping Changes:
```python
# DDP
model = GPTModel(config).to(rank)
ddp_model = DDP(model, device_ids=[rank])

# SFDP
model = GPTModel(config)
auto_wrap_policy = transformer_auto_wrap_policy.transformer_auto_wrap_policy(
    transformer_layer_cls={Block}
)
fsdp_model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    device_id=rank,
)
```

## When to Use SFDP vs DDP

### Use SFDP when:
- ✅ Model is too large for single GPU memory
- ✅ You want to train larger models with same hardware
- ✅ Memory is the bottleneck
- ✅ Training very large language models (>1B parameters)

### Use DDP when:
- ✅ Model fits comfortably on each GPU
- ✅ Communication overhead is a concern
- ✅ Simpler debugging is preferred
- ✅ Smaller models where memory isn't constrained

## Advanced SFDP Features

### 1. CPU Offloading
```python
cpu_offload = CPUOffload(offload_params=True)
# Moves parameters to CPU when not in use
```

### 2. Mixed Precision
```python
from torch.distributed.fsdp import MixedPrecision
mixed_precision = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)
```

### 3. Activation Checkpointing
```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

# Wrap transformer blocks with checkpointing
for i, block in enumerate(model.trf_blocks):
    model.trf_blocks[i] = checkpoint_wrapper(
        block, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
```

## Performance Tips

1. **Optimal Sharding**: Use `transformer_auto_wrap_policy` for transformer models
2. **Batch Size**: Can use larger batch sizes due to memory savings
3. **Prefetching**: Enable forward and backward prefetching for better performance
4. **Mixed Precision**: Combine with FP16/BF16 for additional memory savings
5. **Gradient Clipping**: Apply before optimizer step for stability

## Common Issues and Solutions

### Issue: OOM (Out of Memory)
**Solution**: Enable CPU offloading or reduce batch size

### Issue: Slow Training
**Solution**: Enable prefetching and check communication patterns

### Issue: Convergence Problems
**Solution**: Adjust learning rate, enable gradient clipping

## Summary

SFDP enables training models that would be impossible with DDP due to memory constraints. It's essential for large language models and provides significant memory savings at the cost of some communication overhead. The key benefit is being able to train larger models on the same hardware by distributing the model itself across GPUs, not just the data.