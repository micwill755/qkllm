# DeepSeek Block-Level Sharding Implementation Guide

## Overview
This guide provides a step-by-step walkthrough for implementing block-level FSDP sharding for DeepSeek models, focusing on simplicity and MoE compatibility.

## Step 1: Understanding Block-Level Sharding Strategy

### What We're Doing
- Distributing 61 transformer blocks across multiple GPUs
- Each GPU gets a contiguous range of blocks
- All 8 experts within each MoE layer stay together on the same GPU

### Why This Works Well
- `Block` class provides perfect sharding boundaries
- Each block contains: RMSNorm → MHLA → RMSNorm → MoE
- No cross-GPU expert routing needed
- Standard FSDP auto-wrap policies work out-of-the-box

### Memory Benefits
```
Single GPU:     7B parameters (100% memory)
4-GPU FSDP:     1.75B parameters per GPU (25% memory each)
Memory Savings: 75% reduction per GPU
```

## Step 2: Memory Distribution Analysis

### Current Situation (Single GPU)
- 61 blocks × ~115M params per block = ~7B total parameters
- Plus embedding and output head
- All parameters loaded on one GPU = massive memory usage

### With 4-GPU Block-Level Sharding
```
GPU 0: Embedding + Blocks 0-14     (15 blocks + embedding)
GPU 1: Blocks 15-29                (15 blocks)
GPU 2: Blocks 30-44                (15 blocks)
GPU 3: Blocks 45-60 + Output head  (16 blocks + output)
```

### Parameter Distribution
- **Roughly equal:** ~1.75B parameters per GPU
- **Slight imbalance:** GPU 0 and GPU 3 have embedding/output overhead
- **MoE stays local:** All 8 experts per block remain on same GPU

## Step 3: FSDP Wrapping Strategy

### Key Concept
FSDP needs to know where to create "sharding boundaries" in your model hierarchy.

### Your Wrapping Decision
- **Wrap at:** `Block` level only
- **Don't wrap:** Individual components (MHLA, MoE, RMSNorm, Expert)
- **Let FSDP handle:** Parameter sharding automatically within each block

### What This Means
- Each `Block` becomes an independent FSDP unit
- Parameters within a block are sharded across all GPUs when not in use
- During forward pass, FSDP gathers parameters for the currently active block
- After forward pass, parameters are freed/sharded again

## Step 4: Forward Pass Flow

### Data Flow Through Sharded Model

1. **Input Processing**
   - Starts on GPU 0 with token embedding
   - Input tokens → embedding vectors

2. **GPU 0 Processing (Blocks 0-14)**
   - FSDP gathers Block 0 parameters from all GPUs
   - Process through Block 0 (MHLA + MoE with local experts)
   - FSDP frees Block 0 parameters, gathers Block 1 parameters
   - Repeat for Blocks 1-14
   - All MoE routing happens locally (no cross-GPU communication)

3. **Transfer to GPU 1**
   - Activation tensors sent from GPU 0 to GPU 1
   - Only activations transfer, not parameters

4. **GPU 1 Processing (Blocks 15-29)**
   - Same process: gather parameters, compute, free parameters
   - All experts local to GPU 1

5. **Transfer to GPU 2**
   - Activations GPU 1 → GPU 2

6. **GPU 2 Processing (Blocks 30-44)**
   - Same process with local experts

7. **Transfer to GPU 3**
   - Activations GPU 2 → GPU 3

8. **GPU 3 Processing (Blocks 45-60 + Output)**
   - Process final blocks with local experts
   - Apply final RMSNorm and output projection
   - Generate logits

## Step 5: MoE Routing Simplification

### Why Block-Level Sharding is MoE-Friendly

**Expert Locality:**
- All 8 experts in each MoE layer stay on the same GPU
- Top-2 routing happens entirely locally
- No cross-GPU expert communication needed
- Expert load balancing is per-GPU (simpler to track)

**MoE Processing Flow:**
1. Token activations arrive at a block
2. Router computes expert scores (local computation)
3. Top-2 expert selection (local decision)
4. Expert computation (all experts available locally)
5. Weighted combination (local computation)
6. Result passed to next layer or next GPU

**Benefits:**
- **Simplified routing:** No distributed expert selection
- **Reduced communication:** Only activation transfers between GPUs
- **Easier debugging:** Each GPU's MoE is independent
- **Standard FSDP:** No custom MoE-specific FSDP logic needed

## Step 6: Parameter Gathering Mechanics

### FSDP's Automatic Behavior

**Parameter Lifecycle:**
1. **At rest:** Parameters sharded across all GPUs (1/4 size each)
2. **Before forward:** FSDP gathers all parameters to active GPU
3. **During forward:** Full parameters available for computation
4. **After forward:** Parameters freed and re-sharded
5. **Repeat for backward:** Same gather/compute/free cycle

**Memory Efficiency:**
- Only currently active block has full parameters loaded
- All other blocks have sharded parameters (25% size on each GPU)
- Dramatic memory savings with manageable communication overhead

**Communication Pattern:**
```
Block 0 forward:  All GPUs → GPU 0 (parameter gather)
Block 0 compute:  GPU 0 only (local computation)
Block 0 cleanup:  GPU 0 → All GPUs (parameter shard)

Block 15 forward: All GPUs → GPU 1 (parameter gather)
Block 15 compute: GPU 1 only (local computation)
Block 15 cleanup: GPU 1 → All GPUs (parameter shard)
```

## Step 7: Gradient Synchronization

### Backward Pass Flow

**Gradient Computation:**
- Backward pass happens in reverse order: GPU 3 → GPU 2 → GPU 1 → GPU 0
- Each GPU computes gradients for its assigned blocks
- FSDP automatically handles gradient gathering and reduction

**Expert Gradient Handling:**
- Expert gradients computed locally within each block
- No special cross-GPU gradient communication needed
- Standard FSDP gradient reduction handles everything automatically

**Gradient Flow:**
1. **Loss computation:** On GPU 3 (where logits are generated)
2. **GPU 3 backward:** Gradients for Blocks 45-60 + output
3. **GPU 2 backward:** Gradients for Blocks 30-44
4. **GPU 1 backward:** Gradients for Blocks 15-29
5. **GPU 0 backward:** Gradients for Blocks 0-14 + embedding
6. **All-reduce:** FSDP synchronizes gradients across all GPUs
7. **Parameter update:** Each GPU updates its sharded parameters

## Step 8: Load Balancing Considerations

### What to Monitor

**Expert Utilization:**
- Track which experts are selected most frequently
- Monitor per-GPU expert usage patterns
- Watch for expert load imbalance across GPUs

**Potential Issues:**
- Some GPUs might have blocks with more "popular" experts
- Uneven expert usage could cause computational load imbalance
- Different blocks might have different expert preferences

### Simple Solutions

**For Now (Phase 2):**
- Monitor expert usage per GPU using simple counters
- Add auxiliary loss to encourage balanced expert usage within each block
- Log expert selection statistics during training

**For Later (Phase 3+):**
- Implement dynamic expert load balancing
- Consider expert migration between GPUs
- Advanced routing strategies that consider GPU utilization

## Step 9: Communication Patterns

### Primary Communication Types

**Forward Pass:**
- **Activation transfer:** GPU 0 → GPU 1 → GPU 2 → GPU 3
- **Parameter gathering:** All GPUs → Active GPU (for each block)
- **Parameter freeing:** Active GPU → All GPUs (after each block)

**Backward Pass:**
- **Gradient transfer:** GPU 3 → GPU 2 → GPU 1 → GPU 0
- **Parameter gathering:** All GPUs → Active GPU (for gradient computation)
- **Gradient reduction:** All-reduce across all GPUs

**No MoE-Specific Communication:**
- Expert routing is entirely local
- No cross-GPU expert calls needed
- Standard transformer communication patterns apply

### Communication Efficiency
```
Communication Volume:
- Activations: Small (batch_size × seq_len × emb_dim)
- Parameters: Large but amortized across sequence
- Gradients: Same as parameters, but all-reduced

Total Communication: Much less than expert-level sharding
```

## Step 10: Implementation Requirements

### What You Need to Add

**Distributed Setup:**
- Initialize PyTorch distributed process group
- Set up device ranks and world size
- Configure NCCL backend for GPU communication

**FSDP Integration:**
- Import FSDP and related utilities
- Create auto-wrap policy targeting `Block` class
- Wrap your model with FSDP after moving to GPU

**Training Loop Modifications:**
- Handle distributed loss computation
- Use FSDP-compatible optimizer
- Implement proper checkpointing with FSDP state_dict

**Monitoring and Debugging:**
- Add memory usage tracking
- Monitor expert utilization per GPU
- Log communication patterns and timing

### What Stays the Same

**Model Architecture:**
- Your `Block`, `MoE`, `Expert` classes work unchanged
- Forward/backward logic remains identical
- Expert routing logic stays local and simple

**Training Logic:**
- Same loss functions and optimization objectives
- Same learning rate schedules and regularization
- Same data loading and preprocessing

## Implementation Checklist

### Phase 2A: Basic Setup
- [ ] Set up distributed training environment
- [ ] Create FSDP auto-wrap policy for `Block` class
- [ ] Wrap model with FSDP
- [ ] Test single forward/backward pass

### Phase 2B: Training Integration
- [ ] Modify training loop for distributed setup
- [ ] Implement proper loss reduction across GPUs
- [ ] Add FSDP-compatible checkpointing
- [ ] Validate training convergence

### Phase 2C: Monitoring and Optimization
- [ ] Add memory usage monitoring
- [ ] Track expert utilization per GPU
- [ ] Monitor communication overhead
- [ ] Optimize batch sizes and gradient accumulation

## Expected Results

### Memory Usage
- **Target:** ~25% of single-GPU memory per GPU
- **Reality:** Slightly higher due to FSDP overhead and activation storage
- **Benefit:** Ability to train much larger models or use larger batch sizes

### Performance
- **Speed:** Slight overhead due to parameter gathering/scattering
- **Scalability:** Near-linear scaling with additional GPUs
- **Efficiency:** High GPU utilization with proper batch sizing

### Debugging
- **Simplicity:** Each GPU's computation is largely independent
- **Monitoring:** Standard distributed training monitoring applies
- **Troubleshooting:** Easier than expert-level sharding

## Next Steps After Phase 2

Once block-level sharding is working:
1. **Measure performance:** Baseline memory, speed, and expert utilization
2. **Identify bottlenecks:** Communication, load imbalance, or memory issues
3. **Consider optimizations:** Mixed precision, activation checkpointing
4. **Evaluate expert-level sharding:** If more memory efficiency needed

This approach gives you 75% memory reduction with minimal complexity while keeping MoE routing simple and local.