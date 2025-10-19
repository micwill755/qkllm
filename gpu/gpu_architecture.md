# GPU Architecture for Deep Learning: A Complete Guide

## Overview

This guide explains GPU architecture from the perspective of deep learning workloads, focusing on how hardware design impacts LLM training and inference performance.

## GPU vs CPU Architecture

### Fundamental Design Philosophy

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Design Goal** | Low latency, complex control | High throughput, parallel processing |
| **Core Count** | 4-32 cores | 1,000-10,000+ cores |
| **Core Complexity** | Complex, out-of-order execution | Simple, in-order execution |
| **Memory** | Large caches, complex hierarchy | High bandwidth, simpler hierarchy |
| **Workload** | Sequential, branchy code | Parallel, regular computations |

```
CPU Architecture (Intel/AMD)          GPU Architecture (NVIDIA/AMD)
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │   │ ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐ │
│  │Core1│ │Core2│ │Core3│ │Core4│ │   │ │SM1││SM2││SM3││...││...││SMN│ │
│  │     │ │     │ │     │ │     │ │   │ └───┘└───┘└───┘└───┘└───┘└───┘ │
│  └─────┘ └─────┘ └─────┘ └─────┘ │   │                                 │
│  ┌─────────────────────────────┐ │   │ Each SM contains:               │
│  │      Large L3 Cache         │ │   │ • 64-128 CUDA Cores            │
│  │         (32MB+)             │ │   │ • 4 Tensor Cores               │
│  └─────────────────────────────┘ │   │ • Shared Memory (164KB)        │
│  ┌─────────────────────────────┐ │   │ • L1 Cache (128KB)             │
│  │       System RAM            │ │   │                                 │
│  │      (16-128GB)             │ │   │ ┌─────────────────────────────┐ │
│  └─────────────────────────────┘ │   │ │         VRAM                │ │
└─────────────────────────────────┘   │ │      (8-80GB HBM)           │ │
                                      │ └─────────────────────────────┘ │
                                      └─────────────────────────────────┘
```

## GPU Memory Hierarchy

### Memory Types and Characteristics

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Memory Hierarchy                         │
│                   (Fastest to Slowest)                         │
├─────────────────────────────────────────────────────────────────┤
│ Registers        │ 256KB per SM  │ 1 cycle    │ Per-thread     │
├─────────────────────────────────────────────────────────────────┤
│ Shared Memory    │ 164KB per SM  │ 1-32 cycles│ Per-block      │
├─────────────────────────────────────────────────────────────────┤
│ L1 Cache         │ 128KB per SM  │ 1-32 cycles│ Automatic      │
├─────────────────────────────────────────────────────────────────┤
│ L2 Cache         │ 40MB total    │ 32-200 cyc │ Cross-SM       │
├─────────────────────────────────────────────────────────────────┤
│ VRAM (HBM/GDDR)  │ 8-80GB        │ 200-800 cyc│ Global         │
├─────────────────────────────────────────────────────────────────┤
│ System RAM       │ 16-128GB      │ 1000+ cyc  │ Via PCIe       │
└─────────────────────────────────────────────────────────────────┘
```

### SRAM vs VRAM Detailed Comparison

#### SRAM (Static RAM) - On-Chip Memory
```python
# SRAM characteristics
location = "On GPU die, inside Streaming Multiprocessors"
size_per_sm = {
    "registers": "256KB",
    "shared_memory": "164KB", 
    "l1_cache": "128KB"
}
total_sram_a100 = "108 SMs × 548KB = ~59MB"
access_speed = "1-32 GPU clock cycles"
bandwidth = "~19,000 GB/s (estimated)"
purpose = "Active computation workspace"
management = "Hardware + programmer controlled"
```

#### VRAM (Video RAM) - Off-Chip Memory
```python
# VRAM characteristics  
location = "Separate memory chips on GPU board"
size_options = ["8GB", "16GB", "24GB", "40GB", "80GB"]
memory_types = {
    "consumer": "GDDR6/GDDR6X",
    "datacenter": "HBM2/HBM2e/HBM3"
}
access_speed = "200-800 GPU clock cycles"
bandwidth_examples = {
    "RTX_4090": "1008 GB/s",
    "A100": "1555 GB/s", 
    "H100": "3350 GB/s"
}
purpose = "Model weights, activations, KV cache storage"
management = "Programmer controlled"
```

## Modern GPU Architectures

### NVIDIA GPU Generations

#### Ampere Architecture (A100, RTX 30 Series)
```
┌─────────────────────────────────────────────────────────────────┐
│                        A100 GPU Die                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 108 Streaming Multiprocessors               │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │              Single SM (Streaming Multiprocessor)      │ │ │
│  │  │  ┌─────────────────────────────────────────────────────┐ │ │ │
│  │  │  │ 64 CUDA Cores (FP32)                               │ │ │ │
│  │  │  │ 32 CUDA Cores (FP64)                               │ │ │ │
│  │  │  │ 4 Tensor Cores (3rd gen)                           │ │ │ │
│  │  │  │ 164KB Shared Memory                                │ │ │ │
│  │  │  │ 128KB L1 Cache                                     │ │ │ │
│  │  │  │ 256KB Register File                                │ │ │ │
│  │  │  └─────────────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    40MB L2 Cache                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 5 HBM2e Memory Stacks                      │ │
│  │                    (80GB total)                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### Hopper Architecture (H100)
```
Key Improvements over Ampere:
• 4th Gen Tensor Cores (2x faster for transformers)
• HBM3 memory (3.35 TB/s bandwidth)
• Transformer Engine (FP8 support)
• 50MB L2 Cache (25% larger)
• 132 SMs (22% more than A100)
```

### Memory Bandwidth Evolution

| GPU | Memory Type | Bandwidth | Year |
|-----|-------------|-----------|------|
| V100 | HBM2 | 900 GB/s | 2017 |
| A100 | HBM2e | 1555 GB/s | 2020 |
| H100 | HBM3 | 3350 GB/s | 2022 |
| RTX 4090 | GDDR6X | 1008 GB/s | 2022 |

## GPU Programming Model

### CUDA Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        CUDA Programming Model                   │
├─────────────────────────────────────────────────────────────────┤
│ Grid (Kernel Launch)                                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Block 0    │ Block 1    │ Block 2    │ ...    │ Block N    │ │
│  │ ┌─────────┐│ ┌─────────┐│ ┌─────────┐│        │ ┌─────────┐│ │
│  │ │Thread 0 ││ │Thread 0 ││ │Thread 0 ││        │ │Thread 0 ││ │
│  │ │Thread 1 ││ │Thread 1 ││ │Thread 1 ││        │ │Thread 1 ││ │
│  │ │Thread 2 ││ │Thread 2 ││ │Thread 2 ││        │ │Thread 2 ││ │
│  │ │   ...   ││ │   ...   ││ │   ...   ││        │ │   ...   ││ │
│  │ │Thread 1023││Thread 1023││Thread 1023││      │ │Thread 1023││ │
│  │ └─────────┘│ └─────────┘│ └─────────┘│        │ └─────────┘│ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Hardware Mapping:                                               │
│ • Each Block → One Streaming Multiprocessor (SM)               │
│ • Each Thread → One CUDA Core                                  │
│ • Threads in Block → Share Shared Memory                       │
│ • Blocks → Independent, can run on any SM                      │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Access Patterns

```python
# Efficient GPU memory access patterns
def efficient_patterns():
    # 1. Coalesced Access (Good)
    # Threads access consecutive memory locations
    for thread_id in range(32):  # Warp size
        data[thread_id] = process(input[thread_id])
    
    # 2. Strided Access (Bad)  
    # Threads access memory with large strides
    for thread_id in range(32):
        data[thread_id] = process(input[thread_id * 1000])
    
    # 3. Shared Memory Usage (Good)
    # Load data into fast shared memory first
    shared_data = load_to_shared_memory(global_data)
    for thread_id in range(block_size):
        result[thread_id] = compute(shared_data[thread_id])
```

## Deep Learning Workload Characteristics

### Matrix Operations

```python
# Why GPUs excel at deep learning
def matrix_multiplication_gpu_advantage():
    # CPU approach (sequential)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j] += A[i][k] * B[k][j]  # One operation at a time
    
    # GPU approach (parallel)
    # Launch M×N threads simultaneously
    def gpu_kernel(thread_i, thread_j):
        result = 0
        for k in range(K):
            result += A[thread_i][k] * B[k][thread_j]
        C[thread_i][thread_j] = result
    
    # All M×N operations happen in parallel!
```

### Tensor Cores

Modern GPUs include specialized Tensor Cores for AI workloads:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tensor Core Operation                     │
│                                                                 │
│  Input A (4×4 FP16)    Input B (4×4 FP16)    Accumulator C     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐ │
│  │ a00 a01 a02 a03 │   │ b00 b01 b02 b03 │   │ c00 c01 c02 c03 │ │
│  │ a10 a11 a12 a13 │ × │ b10 b11 b12 b13 │ + │ c10 c11 c12 c13 │ │
│  │ a20 a21 a22 a23 │   │ b20 b21 b22 b23 │   │ c20 c21 c22 c23 │ │
│  │ a30 a31 a32 a33 │   │ b30 b31 b32 b33 │   │ c30 c31 c32 c33 │ │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘ │
│                                    ↓                             │
│                        Single Tensor Core Instruction           │
│                              (1 clock cycle)                    │
│                                    ↓                             │
│                        Output D (4×4 FP32)                      │
│                        ┌─────────────────┐                      │
│                        │ d00 d01 d02 d03 │                      │
│                        │ d10 d11 d12 d13 │                      │
│                        │ d20 d21 d22 d23 │                      │
│                        │ d30 d31 d32 d33 │                      │
│                        └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Characteristics

| Operation | CUDA Cores | Tensor Cores | Speedup |
|-----------|------------|--------------|---------|
| FP32 Matrix Multiply | 19.5 TFLOPS | N/A | 1x |
| FP16 Matrix Multiply | 78 TFLOPS | 312 TFLOPS | 16x |
| INT8 Matrix Multiply | N/A | 624 TOPS | 32x |

## Memory Optimization Strategies

### 1. Flash Attention Memory Usage

```python
# How Flash Attention leverages GPU memory hierarchy
def flash_attention_memory_strategy():
    # Problem: Standard attention needs O(N²) memory
    attention_matrix = allocate_vram(seq_len, seq_len)  # 4096×4096 = 16M elements
    
    # Solution: Flash Attention uses SRAM for computation
    for i in range(0, seq_len, block_size):
        # Load small blocks into fast SRAM
        q_block = load_to_sram(Q[i:i+block_size])      # 64×d_model
        
        for j in range(0, seq_len, block_size):
            k_block = load_to_sram(K[j:j+block_size])  # 64×d_model
            v_block = load_to_sram(V[j:j+block_size])  # 64×d_model
            
            # Compute in SRAM (ultra-fast)
            scores = compute_in_sram(q_block @ k_block.T)  # 64×64 matrix
            # Never store full attention matrix in VRAM!
```

### 2. Paged Attention Memory Layout

```python
# How Paged Attention optimizes VRAM usage
class VRAMLayout:
    def __init__(self):
        # VRAM allocation strategy
        self.vram_sections = {
            "model_weights": "50GB",      # Static, loaded once
            "kv_cache_pool": "25GB",      # Dynamic blocks
            "workspace": "5GB"            # Temporary computations
        }
        
    def allocate_kv_blocks(self):
        # Pre-allocate block pool in VRAM
        total_blocks = 25 * 1024 // 16  # 25GB / 16MB per block
        self.block_pool = allocate_vram(total_blocks, block_size)
        
        # Logical assignment (no VRAM allocation during inference)
        self.block_tables = {}  # seq_id -> [physical_block_ids]
```

## GPU Generations Comparison

### Datacenter GPUs

| GPU | Architecture | VRAM | Memory BW | Tensor Perf | Year |
|-----|-------------|------|-----------|-------------|------|
| V100 | Volta | 32GB HBM2 | 900 GB/s | 125 TFLOPS | 2017 |
| A100 | Ampere | 80GB HBM2e | 1555 GB/s | 312 TFLOPS | 2020 |
| H100 | Hopper | 80GB HBM3 | 3350 GB/s | 1000 TFLOPS | 2022 |

### Consumer GPUs

| GPU | Architecture | VRAM | Memory BW | Use Case |
|-----|-------------|------|-----------|----------|
| RTX 3090 | Ampere | 24GB GDDR6X | 936 GB/s | Small models |
| RTX 4090 | Ada Lovelace | 24GB GDDR6X | 1008 GB/s | Small-medium models |
| RTX 4060 Ti | Ada Lovelace | 16GB GDDR6 | 288 GB/s | Inference only |

## Performance Optimization Guidelines

### Memory Bandwidth Utilization

```python
# Achieving high memory bandwidth utilization
def optimize_memory_bandwidth():
    # 1. Coalesced memory access
    # Good: Sequential access pattern
    for thread_id in range(warp_size):
        data[base_address + thread_id] = value
    
    # 2. Minimize memory transfers
    # Load data once, compute multiple times
    shared_data = load_to_shared_memory(global_data)
    result1 = compute_operation1(shared_data)
    result2 = compute_operation2(shared_data)
    
    # 3. Use appropriate data types
    # FP16 for inference (2x bandwidth vs FP32)
    # INT8 for quantized models (4x bandwidth vs FP32)
```

### Compute Utilization

```python
# Maximizing compute throughput
def optimize_compute():
    # 1. Use Tensor Cores when possible
    # Ensure matrix dimensions are multiples of 8 (FP16) or 16 (INT8)
    
    # 2. Minimize divergent branches
    # Avoid if-statements that cause threads to take different paths
    
    # 3. Overlap computation and memory transfers
    # Use CUDA streams to pipeline operations
```

## Future Trends

### Emerging Technologies

1. **HBM3e Memory**: 5+ TB/s bandwidth
2. **Chiplet Designs**: Multiple GPU dies per package
3. **Near-Memory Computing**: Processing closer to memory
4. **Optical Interconnects**: Faster multi-GPU communication

### Architecture Evolution

```
Current: Monolithic GPU Die
┌─────────────────────────────────┐
│         Single GPU Die          │
│  ┌─────────────────────────────┐ │
│  │    Compute Units (SMs)      │ │
│  └─────────────────────────────┘ │
│  ┌─────────────────────────────┐ │
│  │       Memory Controllers    │ │
│  └─────────────────────────────┘ │
└─────────────────────────────────┘

Future: Chiplet Design
┌─────────────────────────────────┐
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│ │Compute  │ │Compute  │ │Compute  │ │
│ │Chiplet 1│ │Chiplet 2│ │Chiplet 3│ │
│ └─────────┘ └─────────┘ └─────────┘ │
│ ┌─────────────────────────────────┐ │
│ │      Memory/IO Chiplet          │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

## Summary

GPU architecture is specifically designed for the parallel, compute-intensive workloads that characterize deep learning:

1. **Massive Parallelism**: Thousands of simple cores vs dozens of complex CPU cores
2. **High Memory Bandwidth**: Specialized memory (HBM) with TB/s bandwidth
3. **Specialized Compute Units**: Tensor Cores optimized for AI matrix operations
4. **Memory Hierarchy**: Fast SRAM for active computation, large VRAM for data storage

Understanding this architecture is crucial for:
- **Optimizing Memory Usage**: Flash Attention (SRAM) and Paged Attention (VRAM)
- **Maximizing Throughput**: Proper utilization of parallel compute resources
- **Choosing Hardware**: Matching GPU capabilities to workload requirements

The ongoing evolution toward larger memory capacities, higher bandwidth, and specialized AI accelerators continues to enable larger and more capable language models.