# TensorRT-LLM Paged Attention

## What is Paged Attention?

Paged attention manages KV cache in fixed-size blocks (pages) rather than contiguous memory, eliminating fragmentation and enabling dynamic allocation.

**Traditional KV Cache:**
```
Request A: [████████████████████] ← Contiguous, pre-allocated for max length
Request B: [████████████]         ← Wastes memory if shorter than max
```

**Paged KV Cache:**
```
Request A: [Block 1] → [Block 2] → [Block 3] → [Block 4]
Request B: [Block 5] → [Block 6]
           ↑ Each block = fixed size (64-128 tokens)
           ↑ Non-contiguous, allocated on-demand
```

## Key Configuration Parameters

```python
# Engine build configuration
BuildConfig(
    max_batch_size=256,
    max_num_tokens=8192,
    max_seq_len=2048,
    kv_cache_free_gpu_mem_fraction=0.9,  # Use 90% of free GPU memory for KV cache
    enable_paged_kv_cache=True,          # Enable paged attention
    tokens_per_block=64                   # Block size (typically 64 or 128)
)
```

### kv_cache_free_gpu_mem_fraction
- Controls what percentage of free GPU memory is allocated to KV cache pool
- Default: 0.9 (90%)
- Higher values = more sequences in flight, but less headroom for other operations

### tokens_per_block
- Size of each KV cache block in tokens
- Typical values: 64 or 128 tokens
- Larger blocks = less overhead, but potentially more waste per block

## Memory Allocation

### Pre-allocated Pool
```
Total KV Cache Pool: [Block 0][Block 1][Block 2]...[Block N]
                      ↑ Pre-allocated at engine build time
                      ↑ Size determined by kv_cache_free_gpu_mem_fraction

Sequence A: Uses blocks [0, 1, 2]
Sequence B: Uses blocks [3, 4]
Sequence C: Uses blocks [5, 6, 7, 8]
```

### Runtime Behavior
- Batch manager assigns blocks from pool as sequences grow
- Blocks freed immediately when sequences complete
- Pool is reused across all requests

## Key Benefits

### 1. Eliminates Memory Fragmentation
- No wasted memory from over-allocation
- Blocks can be non-contiguous in physical memory
- Only allocate blocks as sequences actually grow

### 2. Dynamic Growth
- Sequences don't reserve max_seq_len upfront
- Blocks allocated incrementally during generation
- Better memory utilization across varying sequence lengths

### 3. Block Sharing
- Multiple sequences can share blocks (e.g., common prompts)
- Useful for beam search and parallel sampling
- Copy-on-write semantics for shared blocks

### 4. Predictable Memory Usage
- Pre-allocated pool prevents OOM surprises
- Clear memory budget at engine build time
- Easier capacity planning

## Memory Savings Example

**Scenario:**
- 100 sequences
- max_seq_len = 2048 tokens
- Actual average length = 512 tokens

**Traditional approach:**
```
Memory = 100 × 2048 = 204,800 tokens allocated
```

**Paged attention:**
```
Memory = 100 × 512 = 51,200 tokens allocated
Savings = 4x reduction
```

## Integration with TensorRT

### Optimized Kernels
- Paged attention kernels fused into TensorRT engine
- Optimized specifically for NVIDIA GPUs
- Tight integration with CUDA graphs for minimal overhead

### Block Table Management
- Each sequence has a block table (list of block pointers)
- Attention kernel reads from multiple blocks using block table
- Slightly more complex than contiguous access, but highly optimized

### Static vs Dynamic
- More static than pure dynamic approaches
- Requires engine rebuild to change configuration
- Trade-off: less flexibility for better performance

## Best Practices

### Tuning kv_cache_free_gpu_mem_fraction
- Start with 0.9 (90%)
- Reduce if you need memory for other operations
- Increase to 0.95 for maximum throughput if GPU is dedicated

### Choosing tokens_per_block
- 64 tokens: Better for shorter sequences, less waste
- 128 tokens: Better for longer sequences, less overhead
- Profile your workload to determine optimal size

### Monitoring
- Track KV cache utilization during serving
- If frequently running out of blocks, increase kv_cache_free_gpu_mem_fraction
- If utilization is low, can reduce to free memory for other uses

## What is Stored in KV Cache?

### Stored: K (Keys) and V (Values)

The KV cache stores **K (Key) and V (Value)** matrices for all previous tokens, NOT Q (Query).

**Stored in cache:**
- **K (Keys)**: For all previous tokens
- **V (Values)**: For all previous tokens

**NOT stored:**
- **Q (Query)**: Only computed for the current token being generated

### Why This Works

**Attention mechanism:**
```python
# Simplified attention
Q = current_token @ W_q        # Query for NEW token only
K = all_tokens @ W_k           # Keys for ALL tokens (cached)
V = all_tokens @ W_v           # Values for ALL tokens (cached)

attention_scores = Q @ K.T     # [1, seq_len]
attention_weights = softmax(attention_scores)
output = attention_weights @ V # [1, hidden_dim]
```

### During Generation

**Prefill phase (first token):**
```
Input: "The cat sat on"
Q = compute for "on"
K = compute for ["The", "cat", "sat", "on"] → CACHE
V = compute for ["The", "cat", "sat", "on"] → CACHE
```

**Decode phase (subsequent tokens):**
```
Generate token: "the"
Q = compute for "the" (new)
K = retrieve cached + append K for "the" → UPDATE CACHE
V = retrieve cached + append V for "the" → UPDATE CACHE
```

### Memory Per Token in KV Cache

```
Per token per layer:
K: [num_heads, head_dim] or [hidden_dim]
V: [num_heads, head_dim] or [hidden_dim]

Total = 2 × hidden_dim per token per layer
```

**Example (Llama-2 7B):**
- hidden_dim = 4096
- num_layers = 32
- Per token: 2 × 4096 × 32 = 262,144 values
- With FP16: 512 KB per token

### Why Q is Not Cached

**Q is only needed for the current token:**
- Computed on-the-fly for new token
- Used immediately in attention
- Discarded after computing output
- No need to store for future tokens

**K and V are needed for all past tokens:**
- Every new token attends to ALL previous tokens
- Must keep K and V for entire sequence
- Reused at every decode step

### Paged Attention Block Storage

Each block in paged attention stores:
```
Block (64 tokens):
├── K: [64, num_layers, hidden_dim]
└── V: [64, num_layers, hidden_dim]
```

The block table points to where these K and V blocks are stored in GPU memory.

## Key Takeaway

Paged attention in TensorRT-LLM provides efficient, non-contiguous KV cache management that dramatically improves memory utilization and throughput, with tight GPU integration for optimal performance.
