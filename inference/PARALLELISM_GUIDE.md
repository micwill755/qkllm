The key differences between the three parallelism strategies:

Model Parallelism (Vertical Split)
What: Split layers across GPUs
Example: GPU0 has layers 1-6, GPU1 has layers 7-12
Flow: Sequential - data flows GPU0 → GPU1 → GPU2
Best for: When model doesn't fit on single GPU

Tensor Parallelism (Horizontal Split)
What: Split operations within each layer
Example: Each GPU handles 2 attention heads out of 8 total
Flow: Parallel - all GPUs work on same layer simultaneously
Best for: Maximum compute efficiency

Pipeline Parallelism (Batch Split)
What: Different GPUs process different batches/microbatches
Example: GPU0 processes batch1 while GPU1 processes batch2
Flow: Overlapped - like an assembly line
Best for: High throughput serving

Real-World Usage:
Small models (7B): Single GPU, no parallelism needed
Medium models (13-30B): Model parallelism
Large models (70B+): Tensor + Pipeline parallelism
Massive models (175B+): All three combined (3D parallelism)

For Your Learning:
Start with understanding model parallelism - it's the simplest and most intuitive. Then explore tensor parallelism when you want to understand how modern systems like vLLM achieve high performance.

The guide shows practical code examples for each approach so you can see exactly how they work under the hood!

# Model vs Tensor vs Pipeline Parallelism

## Visual Overview

```
Single GPU (No Parallelism):
┌─────────────────────────────────────┐
│ GPU 0: Full Model (All Layers)     │
│ Input → Layer1 → Layer2 → ... → Out│
└─────────────────────────────────────┘

Model Parallelism (Layer-wise split):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ GPU 0       │    │ GPU 1       │    │ GPU 2       │
│ Layer 1-4   │ →  │ Layer 5-8   │ →  │ Layer 9-12  │
└─────────────┘    └─────────────┘    └─────────────┘

Tensor Parallelism (Operation-wise split):
┌─────────────────────────────────────────────────────┐
│              Same Layer Across GPUs                 │
│ ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│ │ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │ │
│ │Head 1-2 │  │Head 3-4 │  │Head 5-6 │  │Head 7-8 │ │
│ └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
└─────────────────────────────────────────────────────┘

Pipeline Parallelism (Batch-wise split):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ GPU 0       │    │ GPU 1       │    │ GPU 2       │
│ Batch 1     │    │ Batch 2     │    │ Batch 3     │
│ Layer 1-4   │    │ Layer 5-8   │    │ Layer 9-12  │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 1. Model Parallelism (Layer Parallelism)

**What:** Split model layers across different GPUs
**When:** Model too large for single GPU memory

### Example Implementation:
```python
class ModelParallelGPT:
    def __init__(self, layers, num_gpus):
        self.num_gpus = num_gpus
        layers_per_gpu = len(layers) // num_gpus
        
        # Distribute layers across GPUs
        self.gpu_layers = {}
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * layers_per_gpu
            end_idx = start_idx + layers_per_gpu
            
            # Move layers to specific GPU
            self.gpu_layers[gpu_id] = layers[start_idx:end_idx].to(f'cuda:{gpu_id}')
    
    def forward(self, x):
        # Sequential processing across GPUs
        for gpu_id in range(self.num_gpus):
            # Move input to current GPU
            x = x.to(f'cuda:{gpu_id}')
            
            # Process through layers on this GPU
            for layer in self.gpu_layers[gpu_id]:
                x = layer(x)
        
        return x
```

### Real Example (GPT-3 175B):
```python
# 8 GPUs, each with ~22B parameters
GPU 0: Embedding + Layers 0-23    (22B params)
GPU 1: Layers 24-47               (22B params)  
GPU 2: Layers 48-71               (22B params)
GPU 3: Layers 72-95               (22B params)
GPU 4: Layers 96-119              (22B params)
GPU 5: Layers 120-143             (22B params)
GPU 6: Layers 144-167             (22B params)
GPU 7: Layers 168-191 + Output    (22B params)
```

**Pros:**
- Simple to implement
- Works with any model size
- No changes to model architecture

**Cons:**
- Sequential processing (GPU utilization low)
- High communication overhead
- Bubble time (GPUs waiting)

## 2. Tensor Parallelism (Intra-layer Parallelism)

**What:** Split individual operations (like attention) across GPUs
**When:** Want to parallelize computation within layers

### Attention Tensor Parallelism:
```python
class TensorParallelAttention:
    def __init__(self, d_model, num_heads, world_size):
        self.world_size = world_size
        self.heads_per_gpu = num_heads // world_size
        
        # Each GPU gets subset of attention heads
        self.local_q = Linear(d_model, d_model // world_size)
        self.local_k = Linear(d_model, d_model // world_size) 
        self.local_v = Linear(d_model, d_model // world_size)
        self.output_proj = Linear(d_model, d_model)
    
    def forward(self, x):
        # Each GPU computes its attention heads
        local_q = self.local_q(x)  # Shape: [batch, seq, d_model/world_size]
        local_k = self.local_k(x)
        local_v = self.local_v(x)
        
        # Compute attention for local heads
        local_attn = scaled_dot_product_attention(local_q, local_k, local_v)
        
        # All-gather to combine all heads
        all_attn_outputs = [torch.zeros_like(local_attn) for _ in range(self.world_size)]
        dist.all_gather(all_attn_outputs, local_attn)
        
        # Concatenate all attention heads
        combined_attn = torch.cat(all_attn_outputs, dim=-1)
        
        # Final output projection
        return self.output_proj(combined_attn)
```

### MLP Tensor Parallelism:
```python
class TensorParallelMLP:
    def __init__(self, d_model, d_ff, world_size):
        self.world_size = world_size
        
        # Split the first linear layer column-wise
        self.fc1 = Linear(d_model, d_ff // world_size)  # Each GPU gets part of hidden dim
        
        # Split the second linear layer row-wise  
        self.fc2 = Linear(d_ff // world_size, d_model)
    
    def forward(self, x):
        # Each GPU computes part of the MLP
        local_hidden = F.gelu(self.fc1(x))  # [batch, seq, d_ff/world_size]
        local_output = self.fc2(local_hidden)  # [batch, seq, d_model]
        
        # All-reduce to sum contributions from all GPUs
        dist.all_reduce(local_output, op=dist.ReduceOp.SUM)
        
        return local_output
```

**Pros:**
- All GPUs work simultaneously
- Good GPU utilization
- Scales well with GPU count

**Cons:**
- Requires model architecture changes
- High communication volume
- Complex implementation

## 3. Pipeline Parallelism

**What:** Process different batches/microbatches simultaneously
**When:** Want to improve throughput with multiple requests

### Basic Pipeline:
```python
class PipelineParallelGPT:
    def __init__(self, layers, num_stages, microbatch_size):
        self.num_stages = num_stages
        self.microbatch_size = microbatch_size
        
        # Divide layers into pipeline stages
        layers_per_stage = len(layers) // num_stages
        self.stages = []
        
        for stage_id in range(num_stages):
            start_idx = stage_id * layers_per_stage
            end_idx = start_idx + layers_per_stage
            stage_layers = layers[start_idx:end_idx]
            self.stages.append(stage_layers.to(f'cuda:{stage_id}'))
    
    def forward_pipeline(self, batch):
        # Split batch into microbatches
        microbatches = torch.chunk(batch, batch.size(0) // self.microbatch_size)
        
        # Pipeline execution
        stage_outputs = [[] for _ in range(self.num_stages)]
        
        for step in range(len(microbatches) + self.num_stages - 1):
            for stage_id in range(self.num_stages):
                if step >= stage_id and step - stage_id < len(microbatches):
                    microbatch_idx = step - stage_id
                    
                    if stage_id == 0:
                        # First stage: process input microbatch
                        x = microbatches[microbatch_idx].to(f'cuda:{stage_id}')
                    else:
                        # Later stages: get input from previous stage
                        x = stage_outputs[stage_id-1].pop(0)
                        x = x.to(f'cuda:{stage_id}')
                    
                    # Process through current stage
                    for layer in self.stages[stage_id]:
                        x = layer(x)
                    
                    # Store output for next stage
                    if stage_id < self.num_stages - 1:
                        stage_outputs[stage_id].append(x)
        
        return stage_outputs[-1]  # Final outputs
```

### Advanced Pipeline (GPipe style):
```python
# Timeline visualization:
# Time →
# GPU 0: [MB1] [MB2] [MB3] [MB4] ...
# GPU 1:      [MB1] [MB2] [MB3] [MB4] ...  
# GPU 2:           [MB1] [MB2] [MB3] [MB4] ...
# GPU 3:                [MB1] [MB2] [MB3] [MB4] ...

class GPipeParallel:
    def __init__(self, model, num_microbatches):
        self.num_microbatches = num_microbatches
        self.model = model
    
    def forward(self, batch):
        # Split into microbatches
        microbatches = torch.chunk(batch, self.num_microbatches)
        
        # Forward pass pipeline
        outputs = []
        for microbatch in microbatches:
            output = self.model(microbatch)
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)
```

**Pros:**
- High throughput for multiple requests
- Good GPU utilization
- Works with existing models

**Cons:**
- Adds latency for single requests
- Complex scheduling
- Memory overhead for intermediate activations

## Combining Parallelism Strategies

### 3D Parallelism (All Three Combined):
```python
# Example: 64 GPUs total
# - 8-way tensor parallelism (within each layer)
# - 4-way pipeline parallelism (across layers) 
# - 2-way data parallelism (across batches)

class ThreeDParallelGPT:
    def __init__(self):
        self.tensor_parallel_size = 8    # Split operations
        self.pipeline_parallel_size = 4  # Split layers
        self.data_parallel_size = 2      # Split batches
        
        # Total GPUs = 8 × 4 × 2 = 64 GPUs
```

## Performance Comparison

| Strategy | Memory Efficiency | Compute Efficiency | Communication | Complexity |
|----------|-------------------|-------------------|---------------|------------|
| **Model Parallel** | Excellent | Poor | Low | Low |
| **Tensor Parallel** | Good | Excellent | High | High |
| **Pipeline Parallel** | Good | Good | Medium | Medium |
| **3D Parallel** | Excellent | Excellent | High | Very High |

## When to Use Each

### **Model Parallelism**
- Model doesn't fit on single GPU
- Simple implementation needed
- Latency not critical

### **Tensor Parallelism** 
- Need maximum compute efficiency
- Have high-bandwidth interconnect (NVLink)
- Willing to modify model architecture

### **Pipeline Parallelism**
- Processing many requests simultaneously
- Want to improve throughput
- Have multiple similar-sized requests

### **Combined (3D Parallelism)**
- Extremely large models (100B+ parameters)
- Maximum performance needed
- Have large GPU clusters

## Real-World Examples

### **GPT-3 (175B)**
```python
# OpenAI's approach (estimated)
- Model Parallelism: 8-way (across layers)
- Tensor Parallelism: 4-way (within layers)  
- Data Parallelism: 16-way (across batches)
# Total: 8 × 4 × 16 = 512 GPUs
```

### **PaLM (540B)**
```python
# Google's approach
- Tensor Parallelism: 12-way
- Pipeline Parallelism: 12-way
- Data Parallelism: 64-way
# Total: 12 × 12 × 64 = 9,216 TPUs
```

## Key Takeaway

**Choose based on your constraints:**
- **Memory limited**: Model parallelism
- **Compute limited**: Tensor parallelism  
- **Throughput limited**: Pipeline parallelism
- **All of the above**: 3D parallelism

Most production systems use **tensor + pipeline parallelism** as the sweet spot between complexity and performance.

## Quick Summary

### **Model Parallelism** (Vertical Split)
- **What**: Split layers across GPUs
- **Example**: GPU0 has layers 1-6, GPU1 has layers 7-12
- **Flow**: Sequential - data flows GPU0 → GPU1 → GPU2
- **Best for**: When model doesn't fit on single GPU

### **Tensor Parallelism** (Horizontal Split)  
- **What**: Split operations within each layer
- **Example**: Each GPU handles 2 attention heads out of 8 total
- **Flow**: Parallel - all GPUs work on same layer simultaneously
- **Best for**: Maximum compute efficiency

### **Pipeline Parallelism** (Batch Split)
- **What**: Different GPUs process different batches/microbatches
- **Example**: GPU0 processes batch1 while GPU1 processes batch2
- **Flow**: Overlapped - like an assembly line
- **Best for**: High throughput serving

## Real-World Usage by Model Size

- **Small models (7B)**: Single GPU, no parallelism needed
- **Medium models (13-30B)**: Model parallelism 
- **Large models (70B+)**: Tensor + Pipeline parallelism
- **Massive models (175B+)**: All three combined (3D parallelism)

## Learning Path

1. **Start with Model Parallelism** - Simplest and most intuitive
2. **Understand Tensor Parallelism** - How modern systems like vLLM achieve high performance
3. **Explore Pipeline Parallelism** - For high-throughput serving scenarios
4. **Study 3D Parallelism** - Advanced topic for massive scale deployments