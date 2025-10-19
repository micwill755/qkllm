# Tokens Per Block: Optimizing PagedAttention Block Size

## What is Tokens Per Block?

Tokens per block determines how many tokens are stored in each memory block within PagedAttention's memory pool. This parameter fundamentally affects memory efficiency, latency, and throughput characteristics of LLM inference.

## Block Size Visualization

### Small Blocks (16 tokens per block)
```
Sequence: "The quick brown fox jumps over the lazy dog and runs through the forest"
Total tokens: 15

Block Layout:
┌─────────────────────────────────────────────────────────────┐
│ Block 0: [The|quick|brown|fox|jumps|over|the|lazy|dog|and|  │
│          runs|through|the|forest|<empty>]                   │
│                                                             │
│ Utilization: 14/16 tokens = 87.5%                          │
│ Wasted space: 2 tokens                                     │
└─────────────────────────────────────────────────────────────┘
```

### Large Blocks (64 tokens per block)
```
Sequence: "The quick brown fox jumps over the lazy dog and runs through the forest"
Total tokens: 15

Block Layout:
┌─────────────────────────────────────────────────────────────┐
│ Block 0: [The|quick|brown|fox|jumps|over|the|lazy|dog|and|  │
│          runs|through|the|forest|<empty>|<empty>|<empty>|   │
│          <empty>|<empty>|...|<empty>]                       │
│                                                             │
│ Utilization: 15/64 tokens = 23.4%                          │
│ Wasted space: 49 tokens                                    │
└─────────────────────────────────────────────────────────────┘
```

## Memory Efficiency Analysis

### Internal Fragmentation Comparison

```
Sequence Length vs Block Utilization:

Small Blocks (16 tokens):
Seq Length:  10    25    50    100   200
Blocks Used: 1     2     4     7     13
Utilization: 62%   78%   78%   89%   96%
Avg Waste:   6     6     6     6     8 tokens

Large Blocks (64 tokens):
Seq Length:  10    25    50    100   200
Blocks Used: 1     1     1     2     4
Utilization: 16%   39%   78%   78%   78%
Avg Waste:   54    39    14    28    56 tokens
```

### Memory Pool Efficiency

```
Memory Pool with 1000 blocks:

16 tokens/block:
┌─────────────────────────────────────────────────────────────┐
│ Total capacity: 16,000 tokens                               │
│ Typical utilization: 85-95%                                │
│ Effective capacity: 13,600-15,200 tokens                   │
│ Sequences supported: ~850-950 (avg 16 tokens each)         │
└─────────────────────────────────────────────────────────────┘

64 tokens/block:
┌─────────────────────────────────────────────────────────────┐
│ Total capacity: 64,000 tokens                               │
│ Typical utilization: 60-80%                                │
│ Effective capacity: 38,400-51,200 tokens                   │
│ Sequences supported: ~600-800 (avg 64 tokens each)         │
└─────────────────────────────────────────────────────────────┘
```

## Latency Impact Analysis

### Memory Access Patterns

```
Attention Computation for 200-token sequence:

Small Blocks (16 tokens):
Memory Operations:
├── Fetch Block 0 (tokens 0-15)
├── Fetch Block 1 (tokens 16-31)
├── Fetch Block 2 (tokens 32-47)
├── Fetch Block 3 (tokens 48-63)
├── ... (13 total block fetches)
└── Fetch Block 12 (tokens 192-199)

Total Memory Transfers: 13
Block Table Lookups: 13
Kernel Launches: 13

Large Blocks (64 tokens):
Memory Operations:
├── Fetch Block 0 (tokens 0-63)
├── Fetch Block 1 (tokens 64-127)
└── Fetch Block 2 (tokens 128-199)

Total Memory Transfers: 4
Block Table Lookups: 4
Kernel Launches: 4
```

### Latency Breakdown

```
Per-Token Latency Components:

Small Blocks:
┌─────────────────────────────────────────────────────────────┐
│ Memory Transfer:    ████████ (40ms)                         │
│ Block Lookup:       ██ (10ms)                               │
│ Kernel Overhead:    ████ (20ms)                             │
│ Computation:        ██████ (30ms)                           │
│ Total:              100ms per sequence                      │
└─────────────────────────────────────────────────────────────┘

Large Blocks:
┌─────────────────────────────────────────────────────────────┐
│ Memory Transfer:    ████████████ (60ms)                     │
│ Block Lookup:       █ (5ms)                                 │
│ Kernel Overhead:    ██ (10ms)                               │
│ Computation:        ██████ (30ms)                           │
│ Total:              105ms per sequence                      │
└─────────────────────────────────────────────────────────────┘
```

## Throughput Characteristics

### Batch Processing Efficiency

```
Batch Size Impact on Throughput:

Small Blocks (Better for Large Batches):
Batch Size:     8      16     32     64     128
Memory Usage:   Low    Low    Med    High   Very High
Throughput:     Good   Good   Good   Good   Excellent
Bottleneck:     Compute Compute Memory Memory Memory

Large Blocks (Better for Small Batches):
Batch Size:     8      16     32     64     128
Memory Usage:   Med    High   High   Very High Extreme
Throughput:     Excellent Good  Fair   Poor   Poor
Bottleneck:     Compute Memory Memory Memory Memory
```

### Workload-Specific Performance

```
Performance by Use Case:

Chat Applications (short sequences, high concurrency):
┌─────────────────────────────────────────────────────────────┐
│ Small Blocks (16):                                          │
│ ├── Memory efficiency: Excellent                            │
│ ├── Batch size: Large (128+ sequences)                     │
│ ├── Latency: Good                                           │
│ └── Throughput: Excellent                                   │
│                                                             │
│ Large Blocks (64):                                          │
│ ├── Memory efficiency: Poor                                 │
│ ├── Batch size: Small (16-32 sequences)                    │
│ ├── Latency: Fair                                           │
│ └── Throughput: Good                                        │
└─────────────────────────────────────────────────────────────┘

Document Processing (long sequences, low concurrency):
┌─────────────────────────────────────────────────────────────┐
│ Small Blocks (16):                                          │
│ ├── Memory efficiency: Good                                 │
│ ├── Batch size: Medium (32-64 sequences)                   │
│ ├── Latency: Poor (many blocks)                            │
│ └── Throughput: Good                                        │
│                                                             │
│ Large Blocks (64):                                          │
│ ├── Memory efficiency: Good                                 │
│ ├── Batch size: Small (8-16 sequences)                     │
│ ├── Latency: Excellent (few blocks)                        │
│ └── Throughput: Excellent                                   │
└─────────────────────────────────────────────────────────────┘
```

## Memory Bandwidth Utilization

### Data Transfer Efficiency

```
Memory Bandwidth Usage Pattern:

Small Blocks:
Transfer Size: 16 tokens × 4 bytes × 32 heads × 128 head_dim = 262KB per block
┌─────────────────────────────────────────────────────────────┐
│ GPU Memory Bus (1000 GB/s):                                │
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Utilization: ~8% (many small transfers)                    │
│ Overhead: High (setup cost per transfer)                   │
└─────────────────────────────────────────────────────────────┘

Large Blocks:
Transfer Size: 64 tokens × 4 bytes × 32 heads × 128 head_dim = 1MB per block
┌─────────────────────────────────────────────────────────────┐
│ GPU Memory Bus (1000 GB/s):                                │
│ ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Utilization: ~65% (fewer, larger transfers)                │
│ Overhead: Low (amortized setup cost)                       │
└─────────────────────────────────────────────────────────────┘
```

## Framework Optimization Strategies

### vLLM Approach (16 tokens per block)

```
Design Philosophy: Maximize Serving Throughput
┌─────────────────────────────────────────────────────────────┐
│ Target Workload: Many concurrent short requests             │
│ ├── Chat applications                                       │
│ ├── API serving                                             │
│ └── Interactive applications                                │
│                                                             │
│ Optimization Focus:                                         │
│ ├── Memory efficiency (support more concurrent users)      │
│ ├── Flexible batching (mix short/long sequences)           │
│ └── Continuous batching (add/remove sequences dynamically) │
│                                                             │
│ Trade-offs:                                                 │
│ ├── Higher per-token overhead                               │
│ ├── More complex memory management                          │
│ └── Slightly higher latency for long sequences             │
└─────────────────────────────────────────────────────────────┘
```

### TensorRT-LLM Approach (64+ tokens per block)

```
Design Philosophy: Maximize Raw Performance
┌─────────────────────────────────────────────────────────────┐
│ Target Workload: High-performance inference                 │
│ ├── Batch processing                                        │
│ ├── Long document processing                                │
│ └── Latency-critical applications                           │
│                                                             │
│ Optimization Focus:                                         │
│ ├── Minimize per-token overhead                             │
│ ├── Maximize memory bandwidth utilization                   │
│ └── Reduce kernel launch overhead                           │
│                                                             │
│ Trade-offs:                                                 │
│ ├── Higher memory fragmentation for short sequences        │
│ ├── Lower batch sizes due to memory constraints            │
│ └── Less flexible for mixed workloads                      │
└─────────────────────────────────────────────────────────────┘
```

## Selection Guidelines

### Workload Analysis Matrix

```
Choose Block Size Based on Workload Characteristics:

Sequence Length Distribution:
├── Mostly short (< 100 tokens): Small blocks (16-32)
├── Mixed lengths: Medium blocks (32-48)
└── Mostly long (> 200 tokens): Large blocks (64-128)

Concurrency Requirements:
├── High concurrency (100+ sequences): Small blocks
├── Medium concurrency (10-100 sequences): Medium blocks
└── Low concurrency (< 10 sequences): Large blocks

Latency Sensitivity:
├── Latency critical: Large blocks
├── Balanced: Medium blocks
└── Throughput focused: Small blocks

Memory Constraints:
├── Memory limited: Small blocks (better utilization)
├── Memory abundant: Large blocks (better performance)
└── Balanced: Medium blocks
```

### Performance Tuning Recommendations

```
Optimization Strategy by Use Case:

Real-time Chat:
┌─────────────────────────────────────────────────────────────┐
│ Recommended: 16 tokens per block                            │
│ Rationale:                                                  │
│ ├── Short messages (10-50 tokens typical)                  │
│ ├── High concurrency (100+ users)                          │
│ ├── Memory efficiency critical                              │
│ └── Acceptable latency trade-off                            │
└─────────────────────────────────────────────────────────────┘

Document Summarization:
┌─────────────────────────────────────────────────────────────┐
│ Recommended: 64-128 tokens per block                        │
│ Rationale:                                                  │
│ ├── Long documents (1000+ tokens)                          │
│ ├── Lower concurrency (10-50 documents)                    │
│ ├── Latency sensitive                                       │
│ └── Memory fragmentation acceptable                         │
└─────────────────────────────────────────────────────────────┘

Mixed Workload API:
┌─────────────────────────────────────────────────────────────┐
│ Recommended: 32 tokens per block                            │
│ Rationale:                                                  │
│ ├── Variable sequence lengths                               │
│ ├── Balanced concurrency                                    │
│ ├── Good compromise on efficiency vs performance            │
│ └── Flexible for different request types                    │
└─────────────────────────────────────────────────────────────┘
```

## Performance Monitoring

### Key Metrics to Track

```
Memory Efficiency Metrics:
├── Block utilization percentage
├── Internal fragmentation rate
├── Memory pool exhaustion frequency
└── Average blocks per sequence

Latency Metrics:
├── Time per memory transfer
├── Block lookup overhead
├── Kernel launch frequency
└── End-to-end generation latency

Throughput Metrics:
├── Tokens per second per GPU
├── Concurrent sequences supported
├── Batch processing efficiency
└── Memory bandwidth utilization
```

### Tuning Process

```
Iterative Optimization Approach:

1. Baseline Measurement:
   ├── Measure current performance with default settings
   ├── Identify primary bottleneck (memory vs compute)
   └── Establish performance targets

2. Block Size Experimentation:
   ├── Test 16, 32, 64, 128 tokens per block
   ├── Measure impact on target metrics
   └── Identify optimal range

3. Workload-Specific Tuning:
   ├── Analyze sequence length distribution
   ├── Adjust block size for workload characteristics
   └── Validate performance improvements

4. Production Monitoring:
   ├── Monitor key metrics in production
   ├── Detect performance degradation
   └── Adjust configuration as workload evolves
```

## Summary

The tokens per block parameter represents a fundamental trade-off in PagedAttention systems:

**Small blocks (16-32 tokens)** optimize for memory efficiency and high concurrency, making them ideal for serving many short requests simultaneously. They minimize memory waste but increase per-token processing overhead.

**Large blocks (64-128 tokens)** optimize for raw performance and low latency, making them ideal for processing long sequences or latency-critical applications. They reduce overhead but increase memory fragmentation for short sequences.

**Medium blocks (32-48 tokens)** provide a balanced approach suitable for mixed workloads with variable sequence lengths.

The optimal choice depends on your specific workload characteristics, hardware constraints, and performance priorities. Modern serving frameworks like vLLM and TensorRT-LLM have chosen different defaults based on their target use cases, but both allow tuning for specific requirements.