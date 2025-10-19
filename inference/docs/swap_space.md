# Swap Space in PagedAttention: CPU Memory Overflow Management

## What is Swap Space?

Swap space in vLLM's PagedAttention system refers to CPU RAM allocated as overflow storage when GPU memory becomes full. It acts as a secondary memory tier that allows the system to handle more concurrent sequences than would fit in GPU memory alone.

## Memory Hierarchy with Swap Space

### Two-Tier Memory Architecture

```
Primary Tier (GPU VRAM):
┌─────────────────────────────────────────────────────────────┐
│ GPU Memory Pool (24GB example)                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Block 0    Block 1    Block 2    ...    Block 1499     │ │
│ │ [Active]   [Active]   [Active]          [Active]       │ │
│ └─────────────────────────────────────────────────────────┘ │
│ Status: Fast access (1-2ms latency)                        │
│ Usage: Active sequences, recent sequences                   │
└─────────────────────────────────────────────────────────────┘

Secondary Tier (CPU RAM):
┌─────────────────────────────────────────────────────────────┐
│ CPU Swap Space (2GB example)                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Block 1500 Block 1501 Block 1502 ...  Block 1625      │ │
│ │ [Swapped]  [Swapped]  [Free]           [Free]          │ │
│ └─────────────────────────────────────────────────────────┘ │
│ Status: Slow access (50-100ms latency)                     │
│ Usage: Inactive sequences, overflow storage                 │
└─────────────────────────────────────────────────────────────┘
```

## Memory Allocation Flow

### Normal Operation (GPU Memory Available)

```
Request Flow with Available GPU Memory:

New Request Arrives
        │
        ▼
┌─────────────────┐
│ Check GPU Pool  │
│ Status: 60% full│
└─────────────────┘
        │
        ▼ (GPU space available)
┌─────────────────┐
│ Allocate GPU    │
│ Blocks          │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Begin Processing│
│ (Fast path)     │
└─────────────────┘
```

### Overflow Operation (GPU Memory Full)

```
Request Flow with Full GPU Memory:

New Request Arrives
        │
        ▼
┌─────────────────┐
│ Check GPU Pool  │
│ Status: 100% full│
└─────────────────┘
        │
        ▼ (GPU full)
┌─────────────────┐
│ Check Swap Pool │
│ Status: 40% full│
└─────────────────┘
        │
        ▼ (Swap available)
┌─────────────────┐
│ Allocate CPU    │
│ Blocks          │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Begin Processing│
│ (Slow path)     │
└─────────────────┘
```

## Swapping Mechanisms

### Block Migration Patterns

```
GPU to CPU Swapping (Memory Pressure):

GPU Memory State:
┌─────────────────────────────────────────────────────────────┐
│ Active Seq A  │ Active Seq B  │ Idle Seq C   │ Idle Seq D  │
│ [Recent use]  │ [Recent use]  │ [Old]        │ [Oldest]    │
└─────────────────────────────────────────────────────────────┘
                                      │              │
                                      ▼              ▼
                               Swap to CPU    Swap to CPU

CPU Memory State:
┌─────────────────────────────────────────────────────────────┐
│ Seq C Blocks  │ Seq D Blocks  │ Free Space   │ Free Space  │
│ [Swapped]     │ [Swapped]     │              │             │
└─────────────────────────────────────────────────────────────┘

Result: GPU space freed for new active sequences
```

### CPU to GPU Swapping (Sequence Reactivation)

```
Sequence Reactivation Process:

CPU Memory (Sequence C needs attention):
┌─────────────────────────────────────────────────────────────┐
│ Seq C Blocks  │ Seq D Blocks  │ Free Space   │ Free Space  │
│ [Needs GPU]   │ [Stays CPU]   │              │             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼ (Transfer back to GPU)
GPU Memory:
┌─────────────────────────────────────────────────────────────┐
│ Active Seq A  │ Active Seq B  │ Seq C Blocks │ Free Space  │
│ [Active]      │ [Active]      │ [Restored]   │             │
└─────────────────────────────────────────────────────────────┘

Transfer Time: 50-100ms (PCIe bandwidth limited)
```

## Performance Impact Analysis

### Latency Characteristics

```
Access Latency by Memory Tier:

GPU Memory Access:
┌─────────────────────────────────────────────────────────────┐
│ Memory Transfer: ████ (1-2ms)                               │
│ Computation:     ████████████ (10-15ms)                     │
│ Total Latency:   ████████████████ (12-17ms)                 │
└─────────────────────────────────────────────────────────────┘

CPU Memory Access (Swapped):
┌─────────────────────────────────────────────────────────────┐
│ Swap Transfer:   ████████████████████████████████████████   │
│                  (50-100ms PCIe transfer)                   │
│ Memory Transfer: ████ (1-2ms)                               │
│ Computation:     ████████████ (10-15ms)                     │
│ Total Latency:   ████████████████████████████████████████   │
│                  ████████████████ (62-117ms)                │
└─────────────────────────────────────────────────────────────┘

Performance Impact: 5-10x slower for swapped sequences
```

### Throughput Impact

```
System Throughput with Different Swap Configurations:

No Swap Space (swap_space=0):
┌─────────────────────────────────────────────────────────────┐
│ Concurrent Sequences: 64 (GPU memory limit)                │
│ Request Rejection Rate: High (when GPU full)               │
│ Average Latency: 15ms (all GPU)                            │
│ System Stability: Fail-fast on memory exhaustion          │
└─────────────────────────────────────────────────────────────┘

Small Swap Space (swap_space=2GB):
┌─────────────────────────────────────────────────────────────┐
│ Concurrent Sequences: 80 (64 GPU + 16 CPU)                 │
│ Request Rejection Rate: Medium (when both tiers full)      │
│ Average Latency: 25ms (mixed GPU/CPU)                      │
│ System Stability: Graceful degradation                     │
└─────────────────────────────────────────────────────────────┘

Large Swap Space (swap_space=8GB):
┌─────────────────────────────────────────────────────────────┐
│ Concurrent Sequences: 128 (64 GPU + 64 CPU)                │
│ Request Rejection Rate: Low (large overflow capacity)      │
│ Average Latency: 45ms (many swapped sequences)             │
│ System Stability: High availability, variable performance  │
└─────────────────────────────────────────────────────────────┘
```

## Memory Exhaustion Scenarios

### Complete Memory Pool Exhaustion

```
Memory Exhaustion Timeline:

Time 0: Normal Operation
GPU:  [████████████████░░░░] (80% full)
CPU:  [░░░░░░░░░░░░░░░░░░░░] (0% full)
Status: Accepting requests

Time 1: GPU Memory Full
GPU:  [████████████████████] (100% full)
CPU:  [████░░░░░░░░░░░░░░░░] (20% full)
Status: New requests go to CPU swap

Time 2: Both Tiers Full
GPU:  [████████████████████] (100% full)
CPU:  [████████████████████] (100% full)
Status: Rejecting new requests

Time 3: Sequences Complete
GPU:  [████████████████░░░░] (80% full, some freed)
CPU:  [████████████░░░░░░░░] (60% full, some freed)
Status: Accepting requests again
```

### Request Rejection Behavior

```
Request Handling During Memory Exhaustion:

Incoming Request Processing:
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Check GPU memory availability                       │
│         Result: No free blocks                              │
│                                                             │
│ Step 2: Check CPU swap space availability                   │
│         Result: No free blocks                              │
│                                                             │
│ Step 3: Attempt memory reclamation                          │
│         - Check for completed sequences                     │
│         - Force cleanup of finished requests                │
│         Result: Still no space available                    │
│                                                             │
│ Step 4: Reject request                                      │
│         - Return HTTP 503 Service Unavailable              │
│         - Log memory exhaustion event                       │
│         - Maintain system stability                         │
└─────────────────────────────────────────────────────────────┘
```

## Swap Space Configuration Strategies

### Conservative Configuration (Stability Focus)

```
Configuration: swap_space=2GB (10% of GPU memory)

Memory Distribution:
┌─────────────────────────────────────────────────────────────┐
│ GPU Memory (24GB):                                          │
│ ████████████████████████████████████████████████████████    │
│ Primary storage for active sequences                        │
│                                                             │
│ CPU Swap (2GB):                                             │
│ ████████                                                    │
│ Emergency overflow for peak load                            │
└─────────────────────────────────────────────────────────────┘

Characteristics:
- Low memory overhead
- Minimal performance impact during normal operation
- Limited overflow capacity
- Fast recovery from memory pressure
```

### Balanced Configuration (Flexibility Focus)

```
Configuration: swap_space=6GB (25% of GPU memory)

Memory Distribution:
┌─────────────────────────────────────────────────────────────┐
│ GPU Memory (24GB):                                          │
│ ████████████████████████████████████████████████████████    │
│ Primary storage for active sequences                        │
│                                                             │
│ CPU Swap (6GB):                                             │
│ ████████████████████████                                    │
│ Substantial overflow capacity                                │
└─────────────────────────────────────────────────────────────┘

Characteristics:
- Moderate memory overhead
- Good balance of capacity and performance
- Handles traffic spikes well
- Reasonable latency degradation
```

### Aggressive Configuration (Capacity Focus)

```
Configuration: swap_space=12GB (50% of GPU memory)

Memory Distribution:
┌─────────────────────────────────────────────────────────────┐
│ GPU Memory (24GB):                                          │
│ ████████████████████████████████████████████████████████    │
│ Primary storage for active sequences                        │
│                                                             │
│ CPU Swap (12GB):                                            │
│ ████████████████████████████████████████████████████████    │
│ Large overflow capacity                                      │
└─────────────────────────────────────────────────────────────┘

Characteristics:
- High memory overhead
- Maximum concurrent sequence capacity
- Significant latency impact when swap is used
- Best for high-availability requirements
```

## Monitoring and Optimization

### Key Performance Indicators

```
Memory Utilization Metrics:

GPU Memory Pool:
┌─────────────────────────────────────────────────────────────┐
│ Total Blocks: 1500                                          │
│ Used Blocks:  1200 (80%)                                    │
│ Free Blocks:  300 (20%)                                     │
│ Fragmentation: 5%                                           │
└─────────────────────────────────────────────────────────────┘

CPU Swap Pool:
┌─────────────────────────────────────────────────────────────┐
│ Total Blocks: 125                                           │
│ Used Blocks:  50 (40%)                                      │
│ Free Blocks:  75 (60%)                                      │
│ Swap Events:  15/hour                                       │
└─────────────────────────────────────────────────────────────┘

Performance Metrics:
┌─────────────────────────────────────────────────────────────┐
│ Average Latency: 28ms                                       │
│ GPU-only Requests: 85% (15ms avg)                          │
│ Swapped Requests: 15% (85ms avg)                           │
│ Request Rejection Rate: 2%                                  │
└─────────────────────────────────────────────────────────────┘
```

### Optimization Guidelines

```
Tuning Process:

1. Baseline Measurement:
   ├── Monitor GPU memory utilization patterns
   ├── Track request rejection rates
   ├── Measure average response latencies
   └── Identify peak load characteristics

2. Swap Space Sizing:
   ├── Start with 10-15% of GPU memory
   ├── Increase if rejection rates are high
   ├── Decrease if swap is rarely used
   └── Balance memory cost vs availability

3. Performance Validation:
   ├── Measure latency distribution
   ├── Monitor swap event frequency
   ├── Track system stability metrics
   └── Validate under peak load conditions

4. Production Monitoring:
   ├── Set alerts for high swap usage
   ├── Monitor memory exhaustion events
   ├── Track performance degradation
   └── Adjust configuration based on usage patterns
```

## Use Case Recommendations

### Interactive Applications (Chat, API)

```
Recommended Configuration: swap_space=2-4GB

Rationale:
┌─────────────────────────────────────────────────────────────┐
│ Request Pattern: Short bursts, variable load               │
│ Latency Sensitivity: High (user-facing)                    │
│ Availability Requirements: Medium                           │
│                                                             │
│ Strategy:                                                   │
│ ├── Small swap space for overflow protection               │
│ ├── Prioritize GPU memory for active requests              │
│ ├── Quick rejection of excess requests                     │
│ └── Maintain consistent low latency                        │
└─────────────────────────────────────────────────────────────┘
```

### Batch Processing (Document Analysis)

```
Recommended Configuration: swap_space=8-12GB

Rationale:
┌─────────────────────────────────────────────────────────────┐
│ Request Pattern: Large batches, predictable load           │
│ Latency Sensitivity: Low (batch processing)                │
│ Availability Requirements: High                             │
│                                                             │
│ Strategy:                                                   │
│ ├── Large swap space for maximum throughput                │
│ ├── Accept higher latency for some requests                │
│ ├── Minimize request rejections                            │
│ └── Optimize for total system throughput                   │
└─────────────────────────────────────────────────────────────┘
```

### High-Availability Services

```
Recommended Configuration: swap_space=6-10GB

Rationale:
┌─────────────────────────────────────────────────────────────┐
│ Request Pattern: Unpredictable spikes                      │
│ Latency Sensitivity: Medium                                 │
│ Availability Requirements: Very High                        │
│                                                             │
│ Strategy:                                                   │
│ ├── Substantial swap space for traffic spikes              │
│ ├── Graceful degradation under load                        │
│ ├── Minimize service interruptions                         │
│ └── Balance performance and availability                    │
└─────────────────────────────────────────────────────────────┘
```

## Summary

Swap space in PagedAttention provides a crucial overflow mechanism that transforms hard memory limits into graceful performance degradation. By allocating CPU RAM as secondary storage, systems can:

**Handle Traffic Spikes**: Accommodate more concurrent requests than GPU memory alone would allow, preventing service outages during peak load.

**Provide Graceful Degradation**: Instead of failing fast when memory is exhausted, the system continues operating with reduced performance for some requests.

**Improve System Stability**: Reduce request rejection rates and provide more predictable service availability.

**Enable Flexible Sizing**: Allow operators to trade memory resources for availability and throughput characteristics.

The key insight is that swap space acts as a buffer against memory pressure, allowing systems to maintain service availability at the cost of increased latency for overflow requests. Proper sizing depends on workload characteristics, latency requirements, and availability targets.