# FSDP Memory Lifecycle and Management

## Overview
Understanding how FSDP manages memory is crucial for successful large model training. This guide explains the complete memory lifecycle, storage locations, and solutions for memory constraints.

## GPU Memory Hierarchy

FSDP parameters are stored in **GPU VRAM (Video RAM)** - the main memory of the GPU.

```
GPU Memory Types:
┌─────────────────────────────────────┐
│ GPU VRAM (Main Memory)              │ ← FSDP parameters stored here
│ - Size: 24GB (A100), 80GB (H100)   │
│ - Speed: ~1.5 TB/s bandwidth       │
│ - Stores: Model params, gradients,  │
│   activations, optimizer states     │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ GPU L2 Cache                        │ ← Temporary during computation
│ - Size: ~40MB                       │
│ - Speed: Very fast                  │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ GPU Registers/L1 Cache              │ ← Active computation only
│ - Size: Very small                  │
│ - Speed: Fastest                    │
└─────────────────────────────────────┘
```

## FSDP Memory Lifecycle

### Complete Memory Lifecycle for GPU0

```
1. Initial State (Permanent Storage):
┌─────────────────────────────────────┐
│ Your Shard: Layer3_part1 (5GB)     │
│ Other layers: Layer2_part1, etc.   │
│ Gradients: (5GB)                   │
│ Optimizer states: (3GB)            │
│ Free space: (11GB available)       │
└─────────────────────────────────────┘

2. During Gather (Peak Memory):
┌─────────────────────────────────────┐
│ Your Shard: Layer3_part1 (5GB)     │ ← Still there
│ Received: Layer3_part2 (5GB)       │ ← From GPU1
│ Received: Layer3_part3 (5GB)       │ ← From GPU2  
│ Received: Layer3_part4 (5GB)       │ ← From GPU3
│ Other data: gradients, optimizer... │
│ Free space: (reduced to ~1GB)      │
└─────────────────────────────────────┘

3. After Computation (Back to Base):
┌─────────────────────────────────────┐
│ Your Shard: Layer3_part1 (5GB)     │ ← Updated with gradients
│ Other layers: Layer2_part1, etc.   │
│ Gradients: (5GB)                   │
│ Optimizer states: (3GB)            │
│ Free space: (11GB available)       │ ← Borrowed pieces discarded
└─────────────────────────────────────┘
```

### Step-by-Step Process

**Phase 1: Parameter Gathering**
```python
# GPU0 initiates gather for Layer3
# All GPUs participate simultaneously

GPU0_action = "Request Layer3 pieces from GPU1, GPU2, GPU3"
GPU1_action = "Send Layer3_part2 to GPU0, receive others' pieces"
GPU2_action = "Send Layer3_part3 to GPU0, receive others' pieces" 
GPU3_action = "Send Layer3_part4 to GPU0, receive others' pieces"

# Result: All GPUs have complete Layer3 parameters in VRAM
```

**Phase 2: Gradient Computation**
```python
# Each GPU computes gradients using complete parameters
# But only calculates gradients for their assigned data batch

complete_layer3 = [part1, part2, part3, part4]  # 20GB in VRAM
gradients = compute_backward(complete_layer3, local_batch)
local_gradient = gradients[my_shard_index]  # Only keep my portion
```

**Phase 3: Parameter Update & Cleanup**
```python
# Update only your shard
Layer3_part1 += learning_rate * local_gradient

# Discard borrowed pieces immediately
del Layer3_part2, Layer3_part3, Layer3_part4  # Free 15GB VRAM

# Keep only your updated shard
Layer3_part1  # 5GB in VRAM
```

## Memory Allocation Details

### VRAM Contents During Training
```python
# When you see this in nvidia-smi:
# GPU 0: 45GB / 80GB VRAM used

VRAM_Contents = {
    "model_parameters": "15GB",      # Your 1/N shard of all layers
    "gradients": "15GB",             # Gradients for your shard
    "optimizer_states": "10GB",      # Adam momentum, variance, etc.
    "activations": "3GB",            # Forward pass intermediate values
    "temporary_gathered_params": "2GB"  # During gather operations
}
```

### Memory Timeline
```
Time 1: Base memory = Your shards (25% of model)
Time 2: Peak memory = Complete layer (100% of single layer) 
Time 3: Computation using complete parameters
Time 4: Back to base memory (discard borrowed pieces)
Time 5: Repeat for next layer
```

## When Memory Overflows

### The Problem
```
GPU0 VRAM (24GB total):
┌─────────────────────────────────────┐
│ Your shards: 8GB                    │
│ Gradients: 8GB                      │  
│ Optimizer states: 6GB               │
│ Available: 2GB                      │ ← Not enough for 10GB layer!
└─────────────────────────────────────┘

During gather: Need 10GB for complete layer
Result: RuntimeError: CUDA out of memory
```

### Solution 1: CPU Offloading
```python
cpu_offload=CPUOffload(offload_params=True)
```

**How it works:**
```
Before gather:
GPU VRAM: [Layer3_part1] + [other active data]
CPU RAM:  [Layer1_part1, Layer2_part1] (unused layers)

During Layer3 gather:
GPU VRAM: [Complete Layer3] + [gradients, optimizer states]
CPU RAM:  [Layer1_part1, Layer2_part1] (still unused)

Benefit: More GPU VRAM available for gather operations
Trade-off: CPU↔GPU transfer overhead when layers are needed
```

### Solution 2: Activation Checkpointing
```python
model = checkpoint_wrapper(model)
```

**Memory savings:**
```
Without checkpointing:
VRAM: [parameters] + [all forward activations] + [gradients]

With checkpointing:
VRAM: [parameters] + [minimal activations] + [gradients]
Savings: ~70% reduction in activation memory
Trade-off: ~33% more computation (recompute activations)
```

### Solution 3: Mixed Precision
```python
mixed_precision=MixedPrecision(param_dtype=torch.bfloat16)
```

**Memory reduction:**
```
FP32: Each parameter = 4 bytes
BF16: Each parameter = 2 bytes
Savings: 50% reduction in parameter memory
```

### Solution 4: More GPUs
```python
# 4 GPUs → 8 GPUs
# Each GPU stores 1/8 instead of 1/4 of parameters
# Smaller base memory = more room for gather

4 GPUs: Each stores 25% of model
8 GPUs: Each stores 12.5% of model
```

### Solution 5: Combined Approach
```python
# Maximum memory efficiency
fsdp_model = FSDP(
    checkpoint_wrapper(model),
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    # ... other parameters
)
```

## Memory Planning Guidelines

### Rule of Thumb
```
Required VRAM ≥ largest_layer_size + base_memory_usage

Where:
- largest_layer_size = biggest layer when fully materialized
- base_memory_usage = your shards + gradients + optimizer + activations
```

### Memory Estimation
```python
# For a 70B parameter model on 4 GPUs:
model_size = 70e9 * 4  # 280GB in FP32
per_gpu_shard = 280 / 4  # 70GB per GPU

# With BF16 mixed precision:
per_gpu_shard_bf16 = 70 / 2  # 35GB per GPU

# Largest layer (e.g., attention layer):
largest_layer = 10e9 * 2  # 20GB in BF16

# Required VRAM per GPU:
required_vram = 35 + 20 + 10  # 65GB (shard + largest layer + overhead)
```

### Best Practices

1. **Start conservative**: Use CPU offloading + mixed precision + activation checkpointing
2. **Monitor memory**: Use `torch.cuda.memory_summary()` to track usage
3. **Profile first**: Test with small models to understand memory patterns
4. **Plan for peaks**: Ensure VRAM can handle largest layer materialization
5. **Scale gradually**: Add optimizations incrementally to find the right balance

## Troubleshooting Memory Issues

### Common Error Messages
```
RuntimeError: CUDA out of memory. Tried to allocate 10.00 GiB
```

### Debugging Steps
1. Check current VRAM usage: `nvidia-smi`
2. Identify largest layer: Profile model architecture
3. Calculate required VRAM using formula above
4. Apply memory reduction techniques in order of impact:
   - Mixed precision (50% reduction)
   - CPU offloading (varies)
   - Activation checkpointing (30-70% activation reduction)
   - More GPUs (1/N reduction per GPU)

### Memory Monitoring
```python
import torch

# Track memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Reset peak memory tracking
torch.cuda.reset_peak_memory_stats()
```

The key insight is that FSDP's memory efficiency comes from **temporal management** - parameters are only fully materialized when needed, but this still requires sufficient VRAM to handle the peak memory usage during gather operations.