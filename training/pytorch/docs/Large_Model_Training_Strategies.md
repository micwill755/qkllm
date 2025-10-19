# Training Large Models: Beyond DDP

## The DDP Limitation

DDP requires full model replication on each GPU, making it unsuitable for models that don't fit in single GPU memory.

**Memory Requirements for 670B Parameters:**
- FP16: 1.34 TB per GPU
- FP32: 2.68 TB per GPU
- Current GPU memory: 40-80GB

## Alternative Strategies

### 1. Model Parallelism (Tensor Parallelism)

Split individual layers across GPUs:

```python
# Simplified tensor parallelism example
class DistributedLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        self.world_size = world_size
        self.rank = dist.get_rank()
        
        # Each GPU holds a slice of the weight matrix
        self.weight = nn.Parameter(torch.randn(
            out_features // world_size, in_features
        ))
    
    def forward(self, x):
        # Each GPU computes its slice
        local_output = F.linear(x, self.weight)
        
        # Gather results from all GPUs
        output_list = [torch.zeros_like(local_output) for _ in range(self.world_size)]
        dist.all_gather(output_list, local_output)
        
        return torch.cat(output_list, dim=-1)
```

**Pros**: Can handle any model size
**Cons**: High communication overhead, complex implementation

### 2. Pipeline Parallelism

Split model layers across GPUs sequentially:

```python
# GPU 0: Layers 0-5
# GPU 1: Layers 6-11  
# GPU 2: Layers 12-17
# GPU 3: Layers 18-23

class PipelineGPT(nn.Module):
    def __init__(self, cfg, rank, layers_per_gpu):
        super().__init__()
        start_layer = rank * layers_per_gpu
        end_layer = (rank + 1) * layers_per_gpu
        
        if rank == 0:
            self.embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        self.layers = nn.ModuleList([
            GPT2Block(cfg) for _ in range(layers_per_gpu)
        ])
        
        if rank == 3:  # Last GPU
            self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])
    
    def forward(self, x):
        if hasattr(self, 'embedding'):
            x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        # Send to next GPU
        if self.rank < self.world_size - 1:
            dist.send(x, dst=self.rank + 1)
            return None
        else:
            return self.head(x)
```

**Pros**: Simple to implement, good for sequential models
**Cons**: GPU utilization issues, pipeline bubbles

### 3. ZeRO (Zero Redundancy Optimizer)

DeepSpeed's ZeRO partitions optimizer states, gradients, and parameters:

```python
# Using DeepSpeed ZeRO
import deepspeed

def create_model_and_optimizer(cfg):
    model = GPT2Model(cfg)
    
    # DeepSpeed config
    ds_config = {
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 2,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-4}
        },
        "zero_optimization": {
            "stage": 3,  # Partition parameters, gradients, and optimizer states
            "offload_optimizer": {
                "device": "cpu"  # Offload to CPU memory
            },
            "offload_param": {
                "device": "cpu"
            }
        }
    }
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )
    
    return model_engine, optimizer

# Training loop
def train_with_zero():
    model_engine, optimizer = create_model_and_optimizer(cfg)
    
    for batch in dataloader:
        outputs = model_engine(batch)
        loss = criterion(outputs, targets)
        
        model_engine.backward(loss)
        model_engine.step()
```

**ZeRO Stages:**
- **Stage 1**: Partition optimizer states (4x memory reduction)
- **Stage 2**: Partition gradients (8x memory reduction)  
- **Stage 3**: Partition parameters (memory proportional to model size / num_gpus)

### 4. Fully Sharded Data Parallel (FSDP)

PyTorch's native solution similar to ZeRO:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def create_fsdp_model(cfg):
    model = GPT2Model(cfg)
    
    # Auto-wrap policy for transformer blocks
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GPT2Block}
    )
    
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=True),
    )
    
    return fsdp_model

# Training
fsdp_model = create_fsdp_model(cfg)
optimizer = torch.optim.AdamW(fsdp_model.parameters())

for batch in dataloader:
    outputs = fsdp_model(batch)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 5. Gradient Checkpointing + CPU Offloading

Trade computation for memory:

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList([GPT2Block(cfg) for _ in range(cfg["n_layers"])])
        
    def forward(self, x):
        for block in self.blocks:
            # Recompute activations during backward pass
            x = checkpoint(block, x, use_reentrant=False)
        return x

# CPU offloading for parameters
class CPUOffloadModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Move model to CPU
        self.model.cpu()
    
    def forward(self, x):
        # Move needed parameters to GPU just-in-time
        for name, param in self.model.named_parameters():
            if param.device != x.device:
                param.data = param.data.to(x.device)
        
        output = self.model(x)
        
        # Move parameters back to CPU
        for name, param in self.model.named_parameters():
            param.data = param.data.cpu()
        
        return output
```

## Real-World Solutions for 670B Models

### GPT-3 Style Training (OpenAI)
- **Model Parallelism**: Split attention heads and MLP across GPUs
- **Pipeline Parallelism**: Split layers across GPUs  
- **Data Parallelism**: Multiple model replicas
- **3D Parallelism**: Combines all three approaches

### PaLM/LaMDA Style (Google)
- **Pathways**: Custom distributed training system
- **Sharded parameters**: Similar to ZeRO-3
- **Asynchronous training**: Reduces communication overhead

### Practical Implementation for 670B Model

```python
# Hybrid approach combining multiple strategies
def create_670b_training_setup():
    # 1. Use FSDP for parameter sharding
    # 2. Gradient checkpointing for memory
    # 3. CPU offloading for optimizer states
    # 4. Mixed precision training
    
    model = GPT2Model(cfg_670b)
    
    # Wrap with FSDP
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=True),
        mixed_precision=MixedPrecision(param_dtype=torch.float16),
        activation_checkpointing=True
    )
    
    return fsdp_model

# Estimated requirements for 670B model:
# - 64+ A100 GPUs (40GB each)
# - High-bandwidth interconnect (InfiniBand)
# - Large CPU memory for offloading (1TB+)
# - Specialized frameworks (DeepSpeed, FairScale, or Megatron)
```

## Summary

**DDP is great for models that fit in GPU memory** but fails for 670B+ models because:
1. Requires full model replication
2. Memory scales linearly with number of GPUs
3. No parameter sharding

**For large models, use:**
- **FSDP/ZeRO**: Automatic parameter sharding
- **Model Parallelism**: Manual layer distribution  
- **Pipeline Parallelism**: Sequential layer processing
- **Gradient Checkpointing**: Trade compute for memory
- **CPU Offloading**: Use system RAM as extended GPU memory

The key insight is that modern large model training requires **sharding** (splitting the model) rather than **replication** (copying the model).