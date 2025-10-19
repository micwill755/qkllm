# Dynamic Batching vs Inflight Batching

## Dynamic Batching (Traditional)

### How It Works

- Wait for requests to accumulate up to a timeout or batch size
- Process entire batch together through prefill AND decode
- All requests in batch must finish before new requests can join

```
Batch 1: [Req A, Req B, Req C] → Prefill → Decode → Decode → ... → All finish
Batch 2: [Req D, Req E, Req F] → Prefill → Decode → Decode → ... → All finish
         ↑ waiting while Batch 1 completes
```

### Problems

- **Batch fragmentation**: If Req A finishes early, its slot sits idle while B and C continue
- **Head-of-line blocking**: New requests wait for entire batch to complete
- **Poor GPU utilization**: Wasted compute on empty slots

## Inflight Batching (Continuous Batching)

### How It Works

- Add/remove requests from batch dynamically at each decode step
- When a request finishes, immediately replace it with a new one
- Mix prefill and decode in same batch

```
Step 1: [A_decode, B_decode, C_decode]
Step 2: [A_decode, B_decode, C_decode, D_prefill] ← C finishes, D joins
Step 3: [A_decode, D_decode, E_prefill, F_prefill] ← B finishes, E & F join
```

### Advantages

- **No idle slots**: Continuous GPU utilization
- **Lower latency**: No waiting for batch to complete
- **Higher throughput**: 2-3x improvement in practice
- **Better fairness**: Requests processed as they arrive

## Key Differences

| Aspect | Dynamic Batching | Inflight Batching |
|--------|------------------|-------------------|
| Batch composition | Fixed until all complete | Changes every step |
| GPU utilization | Degrades as requests finish | Stays high |
| Latency | Higher (waiting time) | Lower (immediate scheduling) |
| Throughput | Lower | 2-3x higher |
| Implementation | Simpler | More complex |

## Why Inflight Batching is Superior

### Example Scenario

- Batch of 8 requests
- 4 finish early (short responses)
- **Dynamic batching**: 4 slots idle for remaining time
- **Inflight batching**: 4 new requests immediately fill those slots

### Result

Inflight batching is now the standard for production LLM serving:
- vLLM
- TensorRT-LLM
- Text Generation Inference (TGI)
- All use continuous/inflight batching
