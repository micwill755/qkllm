# Paged Attention: Memory Pool Management for LLM Serving

## What is Paged Attention?

Paged Attention is a memory optimization technique that treats GPU memory like an operating system manages virtual memory. It creates a **shared memory pool** of fixed-size blocks that can be dynamically allocated to different sequences, eliminating the massive memory waste in traditional KV cache management.

## The Problem: KV Cache Memory Waste

### What is KV Cache?

During text generation, LLMs use a KV (Key-Value) cache to avoid recomputing attention for previous tokens. Without KV cache, every new token would require recomputing attention for all previous tokens. With KV cache, only the new token's keys and values need to be computed, then concatenated with cached values.

### The Memory Waste Problem

**Traditional KV Cache Allocation:**
```
VRAM Layout (Wasteful):
┌─────────────────────────────────────────────────────────────┐
│ Seq 1: [████████████░░░░░░░░] (2K used, 2K wasted)         │
│ Seq 2: [██████░░░░░░░░░░░░░░░] (1.5K used, 2.5K wasted)    │
│ Seq 3: [████████████████████] (4K used, 0 wasted)         │
│ Total: 7.5K used, 4.5K wasted (37% waste!)                │
└─────────────────────────────────────────────────────────────┘
```

Each sequence pre-allocates memory for the maximum possible context length, leading to:
- **90%+ memory waste** due to padding unused slots
- **Limited batch sizes** due to over-allocation
- **Poor GPU utilization** despite having available memory

## Paged Attention Solution: Memory Pool Architecture

### Core Concept: Shared Memory Pool

PagedAttention creates a **memory pool** that manages blocks just like an operating system manages virtual memory pages. Instead of per-sequence allocation, all sequences share a common pool of fixed-size blocks.

**Memory Pool Structure:**
```
Memory Pool Status:
┌─────────────────────────────────────────────────────────────┐
│ FREE BLOCKS: [8, 9, 10, 11, 12, 13, 14, 15, ...]          │
│                                                             │
│ USED BLOCKS:                                                │
│ Block 0 → Seq A    Block 4 → Seq C                        │
│ Block 1 → Seq A    Block 5 → Seq C                        │
│ Block 2 → Seq B    Block 6 → Seq C                        │
│ Block 3 → Seq B    Block 7 → Seq D                        │
└─────────────────────────────────────────────────────────────┘
```

### How the Memory Pool Works

**1. Block-based Allocation:**
KV cache is split into fixed-size blocks (typically 16 tokens each). Each block has shape `(num_heads, block_size, head_dim)` and sequences use only the blocks they need.

**2. Virtual-to-Physical Mapping:**
```
Virtual Blocks (logical):     Physical Blocks (VRAM):
Seq 1: [0][1][2]      →      [0]: Seq 1 data
Seq 2: [0][1]         →      [1]: Seq 1 data  
Seq 3: [0][1][2]      →      [2]: Seq 2 data
                              [3]: Seq 2 data
                              [4]: Seq 3 data
                              [5]: Seq 1 data
                              [6]: Seq 3 data
                              [7]: Seq 3 data
```

**3. Efficient Memory Layout:**
```
VRAM Layout (PagedAttention):
┌─────────────────────────────────────────────────────────────┐
│ Block 0: [████████████████] Seq 1 (16 tokens)             │
│ Block 1: [████████████████] Seq 1 (16 tokens)             │
│ Block 2: [████████████████] Seq 2 (16 tokens)             │
│ Block 3: [████████████████] Seq 2 (16 tokens)             │
│ Block 4: [████████████████] Seq 3 (16 tokens)             │
│ Block 5: [████████████████] Seq 1 (16 tokens)             │
│ Block 6: [████████████████] Seq 3 (16 tokens)             │
│ Block 7: [████████████████] Seq 3 (16 tokens)             │
│ Total: 128 tokens used, 0 wasted (0% waste!)              │
└─────────────────────────────────────────────────────────────┘
```

### Dynamic Pool Operations

**When sequence grows:**
- Grab next free block from pool
- Update sequence's block table
- No memory copying required

**When sequence ends:**
- Return all its blocks to free pool
- Blocks immediately available for new sequences

**When sequence is copied (beam search):**
- Just copy the block table pointers
- No actual memory copying needed
- Multiple sequences can share identical prefixes

## Key Benefits

### 1. Memory Efficiency
- **Standard**: 90%+ memory waste due to padding
- **Paged**: Near 100% utilization through sharing

### 2. Higher Throughput
- **Standard**: 3 concurrent requests on 8GB GPU
- **Paged**: 30+ concurrent requests on same GPU

### 3. Dynamic Scaling
- Sequences can grow without pre-allocation
- Memory freed immediately when requests complete

## Paged Attention vs Flash Attention

| Aspect | Flash Attention | Paged Attention |
|--------|----------------|-----------------|
| **Purpose** | Computation optimization | Memory management |
| **Target** | Attention matrix calculation | KV cache storage |
| **Saves** | O(N²) → O(N) computation memory | Eliminates padding waste |
| **When** | During forward/backward pass | During autoregressive generation |
| **Compatibility** | Can be used together | Can be used together |

## Memory Lifecycle

### Request Lifecycle
```python
# Timeline of cache usage
t=0:  Request A starts → allocates blocks [0,1,2]
t=5:  Request B starts → allocates blocks [3,4,5,6] 
t=8:  Request A completes → frees blocks [0,1,2]
t=10: Request C starts → reuses blocks [0,1,2]
t=15: Request B completes → frees blocks [3,4,5,6]
```

### Cache Eviction Strategies

**1. Immediate Cleanup (Most Common)**
```python
def handle_request(prompt):
    request_id = generate_id()
    kv_cache = allocate_cache(request_id)
    
    # Generate response
    for token in generate_response(prompt):
        store_kv(kv_cache, token)
        yield token
    
    # Request completes → immediately free cache
    free_cache(request_id)
```

**2. LRU Eviction (Under Memory Pressure)**
```python
def allocate_blocks(self, request_id, num_blocks):
    if len(self.free_blocks) < num_blocks:
        # Evict oldest completed requests
        self.evict_lru_requests(num_blocks)
    
    return self.free_blocks[:num_blocks]
```

## Real-World Impact

### Before Paged Attention
```
GPU Memory: 8GB
Concurrent Requests: 3 long sequences
Memory Utilization: ~10%
Throughput: Limited by memory waste
```

### After Paged Attention  
```
GPU Memory: 8GB (same)
Concurrent Requests: 30+ mixed sequences
Memory Utilization: ~95%
Throughput: 10x improvement
```

## Advanced Features

### Prefix Caching
Common prefixes (like system prompts) can be cached across requests:

```python
# System prompt cached once, reused across requests
system_prompt = "You are a helpful assistant"
system_kv = cache_prefix(system_prompt)

def new_request(user_prompt):
    kv_cache = copy_prefix(system_kv)  # Reuse cached system tokens
    continue_generation(kv_cache, user_prompt)
```

### Block Sharing
Multiple sequences can share identical prefixes:

```python
# Beam search sharing the same prefix blocks
beam_1: blocks [0,1,2] + [5,6]    # Shared prefix + unique suffix
beam_2: blocks [0,1,2] + [7,8]    # Same prefix, different suffix
```

## Implementation Considerations

### Block Size Selection
- **Small blocks (8-16 tokens)**: Less internal fragmentation, more overhead
- **Large blocks (64-128 tokens)**: More internal fragmentation, less overhead
- **Typical choice**: 16 tokens per block

### Memory Pool Sizing
```python
# Rule of thumb: Size pool for expected peak load
expected_concurrent_requests = 50
avg_sequence_length = 200
safety_factor = 1.5

total_blocks = (expected_concurrent_requests * avg_sequence_length * safety_factor) / block_size
```

## Summary

Paged Attention transforms LLM serving by:

1. **Eliminating Memory Waste**: Shared block pool vs per-sequence allocation
2. **Enabling Higher Throughput**: 10x+ more concurrent requests
3. **Providing Dynamic Scaling**: Memory allocated as needed
4. **Maintaining Compatibility**: Works with Flash Attention and other optimizations

The key insight is treating GPU memory like OS virtual memory - using a shared pool of blocks with dynamic assignment rather than fixed per-process allocation.

This makes Paged Attention essential for production LLM serving where maximizing GPU utilization directly translates to cost savings and improved user experience.