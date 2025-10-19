# FSDP Tuning Parameters Guide

## Overview
FSDP (Fully Sharded Data Parallel) shards model parameters, gradients, and optimizer states across multiple GPUs to enable training of larger models with reduced memory usage per GPU.

## Core Parameters

### Memory Management

#### CPU Offloading
CPU offloading moves model parameters to CPU memory when not actively being used, freeing up GPU memory. This allows training larger models but adds CPU-GPU transfer overhead. Use when GPU memory is the bottleneck.

```python
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

# Keep parameters on GPU (default)
cpu_offload=CPUOffload(offload_params=False)

# Move unused parameters to CPU to save GPU memory
cpu_offload=CPUOffload(offload_params=True)
```

#### Mixed Precision
Mixed precision uses lower precision (16-bit) for parameters and computations instead of 32-bit, reducing memory usage by ~50% and increasing training speed. BFloat16 offers better numerical stability than Float16 but requires newer hardware.

**Parameter Details:**
- **`param_dtype`**: Data type for storing model parameters (weights, biases). Most memory savings come from this setting.
- **`reduce_dtype`**: Data type for gradient reduction across GPUs during all-reduce operations. Affects numerical accuracy of gradient synchronization.
- **`buffer_dtype`**: Data type for non-parameter tensors (batch norm stats, embeddings). Usually small memory impact.

```python
from torch.distributed.fsdp import MixedPrecision

# Aggressive memory saving (may affect stability)
mixed_precision=MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16
)

# Conservative approach (better stability)
mixed_precision=MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,  # Higher precision for gradients
    buffer_dtype=torch.float32   # Keep buffers in full precision
)
```

### Sharding Strategy
Controls how model components are distributed across GPUs. FULL_SHARD provides maximum memory savings by sharding everything. SHARD_GRAD_OP keeps parameters replicated but shards gradients. HYBRID_SHARD balances memory savings with communication efficiency in multi-node setups.

```python
from torch.distributed.fsdp import ShardingStrategy

# Full sharding (default) - shard parameters, gradients, optimizer states
sharding_strategy=ShardingStrategy.FULL_SHARD

# Shard only gradients and optimizer states
sharding_strategy=ShardingStrategy.SHARD_GRAD_OP

# No sharding - behaves like DDP
sharding_strategy=ShardingStrategy.NO_SHARD

# Hybrid sharding - shard within nodes, replicate across nodes
sharding_strategy=ShardingStrategy.HYBRID_SHARD
```

### Auto Wrap Policies

#### Transformer-based Wrapping
Automatically wraps transformer layers for optimal sharding. Each specified layer type becomes a sharding unit, ensuring related parameters stay together and reducing communication overhead during forward/backward passes.

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

# Wrap specific transformer layers
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={Block, AttentionLayer}
)
```

#### Size-based Wrapping
Wraps modules based on parameter count. Modules with fewer parameters than the threshold remain together, reducing sharding overhead for small layers while ensuring large layers are properly distributed.

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# Wrap modules with >= 1M parameters
auto_wrap_policy = partial(
    size_based_auto_wrap_policy, 
    min_num_params=1e6
)
```

### Communication Optimization

#### Backward Prefetch
Overlaps parameter communication with computation during backward pass. BACKWARD_PRE fetches parameters before they're needed, reducing idle time. BACKWARD_POST fetches after computation, using available bandwidth more efficiently.

**How Backward Prefetch Works:**

In FSDP, parameters are sharded across GPUs. During backward pass, each GPU needs to gather parameters from other GPUs before computing gradients.

**What "Gather Parameters" Means:**
"Gather" = collecting all sharded pieces of a layer's parameters from different GPUs to reconstruct the complete layer.

```
Layer 3 Weight Matrix (sharded):
┌─────────────────┐
│ 1  2 │ 3  4    │  ← GPU0 has left half: [1,2,5,6,9,10,13,14]
│ 5  6 │ 7  8    │  ← GPU1 has right half: [3,4,7,8,11,12,15,16]
│ 9 10 │11 12    │
│13 14 │15 16    │
└─────────────────┘

During "Gather":
GPU0 sends its pieces ──┐
                        ├─→ Both GPUs get complete matrix
GPU1 sends its pieces ──┘
```

**Example - 3 Layer Model on 2 GPUs:**

Imagine a model with layers [Layer3, Layer2, Layer1] where:
- GPU0 has: Layer3_left_half, Layer2_left_half, Layer1_left_half
- GPU1 has: Layer3_right_half, Layer2_right_half, Layer1_right_half

Backward pass processes layers in reverse order: Layer3 → Layer2 → Layer1

**1. Without Prefetch (Sequential - SLOW):**
```
Time:  0ms    50ms   100ms  150ms  200ms  250ms  300ms
GPU0:  [Gather Layer3] [Compute Layer3] [Wait...] [Gather Layer2] [Compute Layer2] [Wait...] [Gather Layer1]
       [Get L3 pieces]  [Calc gradients]          [Get L2 pieces]  [Calc gradients]
GPU1:  [Send Layer3]   [Compute Layer3] [Wait...] [Send Layer2]   [Compute Layer2] [Wait...] [Send Layer1]
       [Send L3 pieces] [Calc gradients]          [Send L2 pieces] [Calc gradients]
       
       Problem: GPUs wait idle while gathering (exchanging) parameter pieces!
```

**2. With BACKWARD_PRE (Aggressive - FAST):**
```
Time:  0ms    50ms   100ms  150ms  200ms  250ms
GPU0:  [Gather Layer3] [Compute Layer3] [Compute Layer2] [Compute Layer1]
       [Get L3 pieces]  [Calc L3 grads] [Calc L2 grads]  [Calc L1 grads]
       [Get L2 pieces]  [Get L1 pieces] [Done!]
GPU1:  [Send Layer3]   [Compute Layer3] [Compute Layer2] [Compute Layer1]
       [Send L3 pieces] [Calc L3 grads] [Calc L2 grads]  [Calc L1 grads]
       [Send L2 pieces] [Send L1 pieces] [Done!]
       
       Benefit: While computing Layer3, we're already gathering Layer2 pieces!
```

**3. With BACKWARD_POST (Balanced):**
```
Time:  0ms    50ms   100ms  150ms  200ms  250ms  275ms
GPU0:  [Gather Layer3] [Compute Layer3] [Gather Layer2] [Compute Layer2] [Gather Layer1] [Compute Layer1]
GPU1:  [Send Layer3]   [Compute Layer3] [Send Layer2]   [Compute Layer2] [Send Layer1]   [Compute Layer1]
       
       Benefit: Some overlap, but more conservative memory usage
```

**Why Split Parameters if GPUs Materialize All Parameters?**

The key is **temporal memory savings** - parameters are only materialized temporarily when needed, then discarded.

```
FSDP Memory Lifecycle for GPU0:

1. Permanent Storage: [Layer3_part1] = 25% of layer size

2. During Layer3 backward pass:
   - Gather: Receive parts 2,3,4 from other GPUs
   - Temporary: [Layer3_full] = 100% of layer size (PEAK MEMORY)
   - Compute: Calculate gradients using complete parameters
   - Update: Only Layer3_part1 gets updated
   - Discard: Throw away borrowed parts 2,3,4
   - Back to: [Layer3_part1] = 25% of layer size

3. Move to Layer2 and repeat...
```

**Memory Benefits:**
- **Base memory**: Each GPU stores only 1/N of total model permanently
- **Peak memory**: Only 1 layer's full parameters exist temporarily during computation
- **Result**: Can train 4x larger models than traditional approaches

**Memory Trade-off:**
- **No Prefetch**: Only holds 1 layer's full parameters at a time
- **BACKWARD_PRE**: May hold 2-3 layers' parameters simultaneously (higher memory)
- **BACKWARD_POST**: Holds ~1.5 layers' parameters (moderate memory)

**Key Benefits:**
- **BACKWARD_PRE**: Maximum overlap, fastest training, but uses more memory (holds multiple complete layers)
- **BACKWARD_POST**: Less memory usage, moderate speedup, safer for memory-constrained scenarios
- **None**: No prefetch, slowest but lowest memory usage (only 1 complete layer at a time)

```python
from torch.distributed.fsdp import BackwardPrefetch

# Maximum performance (higher memory usage)
# Use when: You have plenty of GPU memory and want fastest training
backward_prefetch=BackwardPrefetch.BACKWARD_PRE

# Balanced approach (moderate memory usage)
# Use when: Memory is somewhat constrained but you want some speedup
backward_prefetch=BackwardPrefetch.BACKWARD_POST

# No prefetch (lowest memory usage)
# Use when: GPU memory is very tight
backward_prefetch=None
```

#### State Synchronization
`sync_module_states=True` ensures all GPUs start with identical model states, preventing divergence. `use_orig_params=False` allows FSDP to flatten parameters for better memory efficiency, while `True` maintains original parameter structure.

```python
# Synchronize module states across ranks at initialization
sync_module_states=True

# Use original parameter layout (can affect memory usage)
use_orig_params=False
```

## Advanced Optimizations

### Activation Checkpointing
Trades computation for memory by not storing intermediate activations during forward pass. Activations are recomputed during backward pass, reducing memory usage at the cost of ~33% more computation time.

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

# Apply activation checkpointing to save memory
model = checkpoint_wrapper(model)
```

### Gradient Clipping
Prevents gradient explosion by limiting gradient norms to a maximum value. Essential for training stability, especially with large models or high learning rates. The clipping is applied after gradients are synchronized across all GPUs.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Enable gradient clipping
fsdp_model = FSDP(model, ...)

# Clip gradients
torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), max_norm=1.0)
```

## Complete Example

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

# Optimized FSDP configuration
fsdp_model = FSDP(
    model,
    auto_wrap_policy=partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block}
    ),
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    ),
    cpu_offload=CPUOffload(offload_params=True),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    device_id=rank,
    sync_module_states=True,
    use_orig_params=False
)
```

## Performance Tips

- **Start with mixed precision** - Usually provides 30-50% memory savings
- **Enable CPU offloading** for very large models that don't fit in GPU memory
- **Use transformer_auto_wrap_policy** for transformer architectures
- **Enable backward_prefetch** to overlap communication and computation
- **Consider activation checkpointing** for memory-constrained scenarios
- **Monitor GPU memory usage** and adjust parameters accordingly