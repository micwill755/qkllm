# DeepSeek FSDP Training Guide

## Overview
This guide covers distributed training strategies for DeepSeek models using Fully Sharded Data Parallel (FSDP), focusing on the unique challenges of Mixture of Experts (MoE) architectures.

## DeepSeek Architecture Challenges for FSDP

### Current Model Components
- **MoE Layer**: 8 experts with top-2 routing per block
- **MHLA**: Multi-Head Latent Attention with RoPE
- **61 Transformer Blocks**: Pre-norm architecture
- **Custom 1D Arrays**: Manual matrix operations

### Key FSDP Challenges
1. **Expert Distribution**: How to shard 8 experts across multiple GPUs
2. **Dynamic Routing**: Tokens need experts that may be on different GPUs
3. **Load Balancing**: Global expert utilization tracking across shards
4. **Communication Overhead**: MoE requires more inter-GPU communication than standard transformers

## FSDP Sharding Strategies

### Strategy 1: Block-Level Sharding (Simple)
```
GPU 0: Embedding + Blocks 0-14    (~1.75B params)
GPU 1: Blocks 15-29               (~1.75B params)
GPU 2: Blocks 30-44               (~1.75B params)
GPU 3: Blocks 45-60 + Output      (~1.75B params)
```

**Pros:**
- Simple implementation
- Each MoE stays on one GPU
- Standard FSDP auto-wrap policy works

**Cons:**
- Uneven memory (MoE blocks are heavier)
- No expert-level parallelism

### Strategy 2: Expert-Level Sharding (Advanced)
```
GPU 0: All blocks, experts 0,1 in each MoE
GPU 1: All blocks, experts 2,3 in each MoE
GPU 2: All blocks, experts 4,5 in each MoE
GPU 3: All blocks, experts 6,7 in each MoE
```

**Pros:**
- Better load balancing
- True expert parallelism
- More efficient expert utilization

**Cons:**
- Complex routing logic
- Higher communication overhead
- Custom FSDP implementation needed

## MoE Communication Patterns

### Token-to-Expert Routing Options

#### Option A: Send-Compute-Return
```
Token on GPU 0 needs Expert 5 (on GPU 2):
1. Send token embedding to GPU 2
2. GPU 2 computes expert output
3. Send result back to GPU 0
```

#### Option B: All-to-All Expert Gathering
```
1. All GPUs gather expert outputs for all tokens
2. Local routing and weighted combination
3. Higher memory, lower latency
```

#### Option C: Expert Pipeline
```
1. Tokens flow through expert pipeline
2. Each GPU processes its expert subset
3. Results accumulated along pipeline
```

## Memory vs Communication Tradeoffs

### Memory Savings (4 GPUs)
- **Standard Model**: 7B params × 4 = 28B total memory
- **FSDP Sharded**: 7B params ÷ 4 = 1.75B per GPU
- **Memory Reduction**: 75% per GPU

### Communication Overhead
- **Standard Transformer**: Sequential GPU-to-GPU
- **MoE Block-Level**: Same as standard + expert routing
- **MoE Expert-Level**: All-to-all communication for routing

## Implementation Considerations

### PyTorch Conversion Requirements
```python
# Current: class MoE()
# Needed: class MoE(nn.Module)

# Current: self.experts = [Expert(...) for _ in range(8)]
# Needed: self.experts = nn.ModuleList([Expert(...) for _ in range(8)])

# Current: 1D array operations
# Needed: PyTorch tensor operations
```

### FSDP Auto-Wrap Policy
```python
# Block-level sharding
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={Block}
)

# Expert-level sharding (custom)
def expert_wrap_policy(module, recurse, nonwrapped_numel):
    if isinstance(module, Expert):
        return True
    return recurse
```

### Load Balancing Across Shards
```python
# Challenge: Collect expert usage from all GPUs
# Solution: Use distributed reduction for statistics
expert_usage = torch.zeros(num_experts)
dist.all_reduce(expert_usage, op=dist.ReduceOp.SUM)
```

## Performance Optimization Strategies

### 1. Communication Optimization
- **Overlap Computation**: Hide communication with computation
- **Gradient Accumulation**: Reduce communication frequency
- **Mixed Precision**: Use FP16/BF16 for faster transfers

### 2. Expert Utilization
- **Dynamic Load Balancing**: Adjust routing based on GPU utilization
- **Expert Caching**: Cache frequently used experts locally
- **Adaptive Routing**: Learn communication-aware routing

### 3. Memory Management
- **CPU Offloading**: Move unused experts to CPU
- **Activation Checkpointing**: Trade computation for memory
- **Gradient Compression**: Reduce gradient communication size

## Decision Framework

### Choose Block-Level Sharding When:
- ✅ Simpler implementation preferred
- ✅ Standard FSDP features sufficient
- ✅ Communication bandwidth limited
- ✅ Debugging simplicity important

### Choose Expert-Level Sharding When:
- ✅ Maximum memory efficiency needed
- ✅ Expert utilization optimization critical
- ✅ High-bandwidth interconnect available
- ✅ Complex implementation acceptable

## Implementation Roadmap

### Phase 1: PyTorch Conversion
1. Convert 1D arrays to PyTorch tensors
2. Transform classes to nn.Module
3. Implement standard forward/backward passes

### Phase 2: Basic FSDP Integration
1. Implement block-level sharding
2. Test with standard auto-wrap policy
3. Validate training convergence

### Phase 3: MoE Optimization
1. Implement expert-level sharding
2. Optimize communication patterns
3. Add load balancing across GPUs

### Phase 4: Performance Tuning
1. Enable mixed precision
2. Add activation checkpointing
3. Optimize batch sizes and communication

## Key Questions to Consider

1. **Sharding Granularity**: Block-level or expert-level sharding?
2. **Communication Strategy**: How to handle cross-GPU expert access?
3. **Load Balancing**: Global vs local expert utilization tracking?
4. **Implementation Complexity**: Full PyTorch conversion vs hybrid approach?
5. **Performance Priority**: Memory efficiency vs communication overhead?

## Next Steps

1. **Analyze Current Model**: Measure memory usage and bottlenecks
2. **Choose Strategy**: Decide on sharding approach based on requirements
3. **Convert Implementation**: Transform to PyTorch-compatible format
4. **Test Incrementally**: Start with simple sharding, add complexity
5. **Optimize Performance**: Tune communication and memory usage

The key insight is that MoE architectures require careful consideration of expert distribution and routing communication patterns when implementing FSDP, making them more complex than standard transformer models.