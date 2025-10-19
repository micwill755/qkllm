# TensorRT-LLM Memory Constraints

## The Two Key Limitations

### 1. max_batch_size × max_num_tokens Constraint
The total number of tokens you can process is bounded by `max_batch_size * max_num_tokens`. Even if you set `max_num_tokens` very high, you can only process that many tokens if you have enough batch slots available.

### 2. Memory Allocation Trade-off
When you build the TensorRT engine, it pre-allocates memory buffers based on these max values:

- **Larger max_batch_size/max_num_tokens** → More memory for input/activation buffers
- **Less memory left for KV cache** → Fewer tokens can actually be cached
- **Result**: Lower effective throughput despite higher theoretical limits

## Memory Competition

```
Total GPU Memory = Model Weights + Activation Buffers + KV Cache + Overhead
                                    ↑                      ↑
                            (max_batch_size *      (actual runtime
                             max_num_tokens)        token storage)
```

## KV Cache Memory Growth

KV cache memory grows with:
```
batch_size × sequence_length × num_layers × hidden_dim × 2 (K+V)
```

## Practical Implications

- If `max_batch_size` is too large, you allocate huge activation buffers but starve the KV cache
- The KV cache needs to store keys/values for ALL tokens across ALL layers for ALL sequences in flight
- Setting max values too high leads to memory exhaustion and OOM errors

## Best Practice

Find the sweet spot where:
- `max_batch_size` and `max_num_tokens` are large enough for your workload
- But small enough to leave sufficient memory for KV cache to avoid excessive evictions/recomputation

**Recommendation**: Use dynamic batching with conservative max values rather than setting them to theoretical maximums.
