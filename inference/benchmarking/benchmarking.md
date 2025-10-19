# How Do We Measure the Performance of an Inference System?

At the highest level there are two competing metrics:

- **Latency** — the time from when a request is submitted until tokens are returned
- **Throughput** — the number of tokens/requests per second the system can generate/process

**Latency** matters most for interactive applications, where users are waiting on responses.

**Throughput** matters in offline workloads like synthetic data generation for pre/post-training runs, data cleaning/processing, and in general - any type of offline batch inference jobs.

## ASCII Diagram: Request Timeline

```
<-------------- (e2e) latency! -------------->
┌─────────────────────────────────────────────┐
│         vLLM inference server               │
└─────────────────────────────────────────────┘
  ^                |           |            |        
  |  <--- TTFT --->|  <-ITL->  |            | 
  |                |           |            |    
query              |           |            |    
  |                v           v            v    
  |            token 1     token 2     token n
  |                |           |            |    
  |                v           v            v    
┌─────────────────────────────────────────────┐
│                     User                    │
└─────────────────────────────────────────────┘

## Metric Definitions

| Metric | Definition |
|--------|------------|
| **TTFT** (Time to First Token) | Time from request submission until the first output token is received |
| **ITL** (Inter-Token Latency) | Time between two consecutive tokens (e.g., from token i-1 to token i) |
| **TPOT** (Time Per Output Token) | The average ITL across all output tokens in a request |
| **Latency / E2E** (End-to-End Latency) | Total time to process a request, i.e. TTFT + sum of all ITLs, or equivalently the time between submitting request and receiving the last output token |
| **Throughput** | Total tokens processed per second (input, output, or both), or alternatively requests per second |
| **Goodput** | Throughput that meets service-level objectives (SLOs) such as max TTFT, TPOT, or e2e latency. For example, only tokens from requests meeting those SLOs are counted |