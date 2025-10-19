# Chunked Prefill Guide

## What is Chunked Prefill?

**Chunked prefill** is an optimization technique that breaks down the initial prompt processing (prefill phase) into smaller chunks rather than processing the entire prompt at once. The key insight is that this enables **request interleaving** - allowing small requests to be served immediately while large requests are processed gradually.

## The Core Problem

```
Traditional LLM Serving - The "Monopoly" Problem:
┌─────────────────────────────────────────────────────────────┐
│ Request Queue:                                              │
│ 1. "Analyze this 50-page document..." (12,000 tokens)      │
│ 2. "What is 2+2?" (8 tokens)                               │
│ 3. "Hello" (3 tokens)                                      │
│ 4. "Translate: Bonjour" (12 tokens)                        │
└─────────────────────────────────────────────────────────────┘

Processing Timeline:
[────────── Process 12K tokens ──────────] [8 tok] [3 tok] [12 tok]
     ^────── 8 seconds ──────^              ^0.1s^  ^0.1s^  ^0.1s^

Result: Small requests wait 8+ seconds! 
```

## The Chunked Prefill Solution

```
Chunked Prefill - The "Fair Sharing" Solution:
┌─────────────────────────────────────────────────────────────┐
│ Same Request Queue - But Now Processed Differently:        │
│ 1. Large request split into chunks: [chunk1][chunk2]...    │
│ 2-4. Small requests processed immediately                   │
└─────────────────────────────────────────────────────────────┘

Processing Timeline:
[chunk1][8tok][chunk2][3tok][chunk3][12tok][chunk4][chunk5]...
  ^0.5s^ ^0.1s^ ^0.5s^ ^0.1s^ ^0.5s^ ^0.1s^  ^0.5s^  ^0.5s^

Result: Small requests get served immediately! 
```

## Visual Overview: Traditional vs Chunked

```
Traditional Prefill:
┌─────────────────────────────────────────────────────────────┐
│ Input: "The quick brown fox jumps over the lazy dog..."     │
│ Process: [────────── entire sequence ──────────] → token   │
│ Memory:  ████████████████████ (HUGE peak!)                 │
│ Other requests: ⏳ WAITING... ⏳                            │
└─────────────────────────────────────────────────────────────┘

Chunked Prefill:
┌─────────────────────────────────────────────────────────────┐
│ Input: Same sequence                                        │
│ Chunk 1: [The quick brown fox] → partial KV cache          │
│ Memory:  ████ (manageable)                                  │
│ Other requests: ✅ SERVED! ✅                               │
│ Chunk 2: [jumps over the lazy] → update KV cache           │
│ Memory:  ████ (manageable)                                  │
│ Other requests: ✅ SERVED! ✅                               │
│ ... continues until complete                                │
└─────────────────────────────────────────────────────────────┘
```

## The Three Key Benefits

### 1. Memory Efficiency - Taming the O(n²) Beast

```
Attention Memory Usage:

Traditional (8K sequence):
Memory │     ████████████████  ← 8.6GB peak (OOM!)
       │    ██              ██
       │   ██                ██
       │  ██                  ██
       └──────────────────────────→ Time
          ^──── prefill────^

Chunked (512 token chunks):
Memory │ ██  ██  ██  ██  ██  ██  ← 33MB peaks (manageable)
       │██  ██  ██  ██  ██  ██
       │                        
       └──────────────────────────→ Time
         ^chunk^ ^chunk^ ^chunk^

Memory Reduction: 99.6% lower peak usage!
```

**Why this matters:**
- **Attention computation is O(sequence_length²)**
- 8K tokens = 64M attention elements
- 512 tokens = 262K attention elements (256x smaller!)
- Prevents out-of-memory errors on long sequences

### 2. Request Interleaving - The Fairness Revolution

```
Batch Composition Strategy:
┌─────────────────────────────────────────────────────────────┐
│ Smart Scheduler Creates Mixed Batches:                      │
│                                                             │
│ Batch 1: [decode_req1] [decode_req2] [small_prefill] [chunk]│
│          ^─ 1 token ─^ ^─ 1 token ─^ ^─── 50 tokens ──^ ^512^│
│                                                             │
│ Batch 2: [decode_req3] [decode_req4] [small_prefill] [chunk]│
│          ^─ 1 token ─^ ^─ 1 token ─^ ^─── 30 tokens ──^ ^512^│
│                                                             │
│ Result: Everyone gets served quickly!                       │
└─────────────────────────────────────────────────────────────┘

Priority System:
1. 🔥 Decode steps (existing conversations) - 1 token each
2. ⚡ Small prefill requests (new short conversations)
3. 🐌 One chunk from large prefill requests
```

### 3. GPU Utilization - From 45% to 92%

```
GPU Utilization Comparison:

Traditional Serving:
GPU │████████████                    ████                    
    │^─ big req ─^                    ^sm^                    
    │            idle time            req                     
    └─────────────────────────────────────────────────────→ Time
    Utilization: 45% (lots of idle time)

Chunked Prefill Serving:
GPU │████████████████████████████████████████████████████████
    │^chunk^decode^chunk^decode^small^chunk^decode^small^...
    │                                                        
    └─────────────────────────────────────────────────────→ Time
    Utilization: 92% (always busy!)
```

## Real-World Performance Impact

```
Metrics Comparison:
┌─────────────────────────────────────────────────────────────┐
│                    │ Traditional │ Chunked Prefill          │
│────────────────────│─────────────│─────────────────────────│
│ Small Req Latency  │ 8.2 seconds │ 0.15 seconds ✅         │
│ Throughput         │ 15 req/min  │ 85 req/min ✅           │
│ GPU Utilization    │ 45%         │ 92% ✅                  │
│ Memory Efficiency  │ OOM errors  │ Stable ✅               │
│ Request Fairness   │ Poor        │ Excellent ✅            │
└─────────────────────────────────────────────────────────────┘

Result: 5.7x better throughput, 55x lower latency for small requests!
```

## How Modern Systems Use Chunked Prefill

### vLLM's Approach
```
vLLM Scheduling Strategy:
┌─────────────────────────────────────────────────────────────┐
│ 1. Collect requests from queue                              │
│ 2. Fill batch with decode steps first (priority)           │
│ 3. Add small prefill requests                               │
│ 4. Add ONE chunk from large prefill request                │
│ 5. Process mixed batch                                      │
│ 6. Repeat                                                   │
└─────────────────────────────────────────────────────────────┘

Typical Batch (2048 tokens max):
- 8 decode steps (8 tokens)
- 3 small prefill requests (400 tokens)
- 1 large request chunk (512 tokens)
- Remaining space for more decode/small requests
```

### TensorRT-LLM's Approach
```
TensorRT-LLM Strategy:
┌─────────────────────────────────────────────────────────────┐
│ - Adaptive chunk sizing based on available memory          │
│ - KV cache block management for efficient memory reuse     │
│ - Continuous batching with mixed request types             │
│ - Optimized CUDA kernels for chunk processing              │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Guidelines

### Chunk Size Selection
```
Chunk Size Guidelines:
┌─────────────────────────────────────────────────────────────┐
│ Model Size    │ Recommended Chunk Size │ Reasoning          │
│───────────────│────────────────────────│────────────────────│
│ Small (7B)    │ 1024-2048 tokens      │ More memory available│
│ Medium (13-30B)│ 512-1024 tokens      │ Balanced approach   │
│ Large (70B+)  │ 256-512 tokens       │ Memory constrained  │
│ Massive (175B+)│ 128-256 tokens       │ Very tight memory   │
└─────────────────────────────────────────────────────────────┘

Memory Considerations:
- Available GPU memory
- Batch size requirements  
- Sequence length distribution
- Number of concurrent requests
```

### Performance Tuning
```
Optimization Factors:
┌─────────────────────────────────────────────────────────────┐
│ Factor                │ Impact                              │
│───────────────────────│─────────────────────────────────────│
│ Chunk Size            │ Memory vs. scheduling flexibility   │
│ Batch Composition     │ Decode/prefill ratio optimization  │
│ Request Prioritization│ Latency vs. throughput tradeoffs   │
│ Memory Management     │ KV cache efficiency                │
└─────────────────────────────────────────────────────────────┘
```

## Best Practices

### 1. Scheduling Strategy
- **Prioritize decode steps** - they're fast and keep conversations flowing
- **Batch small prefills together** - similar memory characteristics
- **Limit chunks per batch** - usually 1-2 chunks maximum
- **Monitor queue depths** - prevent starvation of large requests

### 2. Memory Management
- **Start conservative** with chunk sizes, then optimize
- **Monitor peak memory usage** during mixed batches
- **Use KV cache pooling** to reduce allocation overhead
- **Consider CPU offloading** for very long sequences

### 3. Request Fairness
- **Set maximum wait times** for large requests
- **Use weighted scheduling** based on request size
- **Implement backpressure** when queues get too long
- **Monitor latency percentiles** across request sizes

## Common Pitfalls

### ❌ What NOT to Do
- **Chunks too small**: Overhead dominates, poor GPU utilization
- **Chunks too large**: Memory spikes, defeats the purpose
- **Ignoring decode priority**: Existing conversations become slow
- **No request mixing**: Back to the original monopoly problem
- **Fixed chunk sizes**: Doesn't adapt to different workloads

### ✅ What TO Do
- **Adaptive chunk sizing** based on available memory
- **Smart batch composition** with mixed request types
- **Continuous monitoring** of latency and throughput metrics
- **Gradual optimization** starting from conservative settings

## The Bottom Line

Chunked prefill transforms LLM serving from a **"first-come, first-served"** system where large requests monopolize resources, into a **"fair time-sharing"** system where everyone gets served quickly.

**Key Insight**: It's not about reducing total computation - it's about enabling **request interleaving** and **memory management** that makes high-throughput, low-latency serving possible.

This is why modern systems like vLLM can serve 100x more requests per second than naive implementations!