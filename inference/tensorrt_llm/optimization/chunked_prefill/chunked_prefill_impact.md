# Chunked Prefill Impact on TensorRT-LLM

## Memory Impact

### Reduces Peak Memory Usage

- **Without chunked prefill**: Process entire prompt (e.g., 10K tokens) in one forward pass → huge activation memory spike
- **With chunked prefill**: Process prompt in chunks (e.g., 512 tokens at a time) → smaller, consistent activation memory

```
Activation Memory ∝ chunk_size (not full prompt length)
```

### Allows More Aggressive Settings

- Can increase `max_batch_size` because each request uses less peak memory
- More memory available for KV cache since activation buffers are smaller
- Better memory utilization across concurrent requests

## Throughput Impact

### Trade-offs

- **Per-request throughput**: Slightly lower (multiple passes vs one pass)
- **TTFT (Time to First Token)**: Higher since prefill takes longer
- **Overall throughput**: Often HIGHER because you can batch more requests simultaneously

## Scheduling Impact

### Without Chunked Prefill: Large Request Starvation

```
Request Queue: [small, small, LARGE, small, small, ...]
                                ↑
                    Needs 10K tokens of contiguous memory
                    but fragmented KV cache blocks it
```

**What happens:**
- Large requests wait for enough contiguous free memory
- Small requests keep getting scheduled first (easier to fit)
- Large request gets perpetually blocked → **starvation**
- Even worse: might OOM and fail entirely

### With Chunked Prefill: Fair Scheduling

- Large request broken into chunks (e.g., 10K tokens → 20 chunks of 512)
- Each chunk has same memory footprint as small requests
- Scheduler can interleave: `[small, LARGE_chunk1, small, LARGE_chunk2, ...]`
- Large request makes incremental progress instead of blocking

**Benefits:**
- Fair scheduling - no request starvation
- Better resource utilization
- Predictable latency bounds (no indefinite waiting)

## Key Takeaway

Chunked prefill is **essential** for production serving with mixed workloads (short + long prompts) because it:
- Reduces peak memory usage
- Enables better resource allocation
- Prevents large request starvation
- Improves overall system throughput
