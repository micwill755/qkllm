# GPT-2 Distributed Data Parallel (DDP) Training Guide

## What is DDP?
Distributed Data Parallel (DDP) allows you to train your model across multiple GPUs simultaneously. Instead of training on one GPU, DDP splits your data across multiple GPUs, trains on each GPU separately, then synchronizes the gradients.

## Step-by-Step Changes Made

### 1. Added Required Imports
```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
```

**What these do:**
- `dist`: Handles communication between GPUs
- `mp`: Creates multiple processes (one per GPU)
- `DDP`: Wraps your model to sync gradients across GPUs
- `DistributedSampler`: Ensures each GPU gets different data batches

### 2. Setup Function
```python
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
```

**What this does:**
- `rank`: Which GPU this process is running on (0, 1, 2, etc.)
- `world_size`: Total number of GPUs
- `MASTER_ADDR/PORT`: How GPUs communicate with each other
- `nccl`: NVIDIA's optimized communication backend for GPUs

### 3. Cleanup Function
```python
def cleanup():
    dist.destroy_process_group()
```

**What this does:**
- Properly shuts down the distributed training when finished

### 4. Distributed DataLoader
```python
def create_dataloader_ddp(txt, batch_size, max_length, stride, rank, world_size):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return dataloader
```

**Key difference from regular DataLoader:**
- `DistributedSampler` ensures each GPU gets different data
- If you have 1000 samples and 2 GPUs:
  - GPU 0 gets samples 0, 2, 4, 6, 8...
  - GPU 1 gets samples 1, 3, 5, 7, 9...

### 5. Main Training Function
```python
def train_ddp(rank, world_size):
    setup(rank, world_size)  # Initialize distributed training
    
    # Load data (same on all GPUs)
    # ... data loading code ...
    
    # Create model and move to specific GPU
    model = GPTModel(GPT_CONFIG_124M).to(rank)
    ddp_model = DDP(model, device_ids=[rank])  # Wrap with DDP
    
    # Create distributed dataloader
    train_loader = create_dataloader_ddp(...)
    
    # Training loop
    for epoch in range(3):
        train_loader.sampler.set_epoch(epoch)  # Important for shuffling
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # ... training code ...
            
            if rank == 0 and batch_idx % 10 == 0:  # Only GPU 0 prints
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    cleanup()  # Clean up when done
```

**Key changes:**
- `model.to(rank)`: Moves model to specific GPU
- `DDP(model, device_ids=[rank])`: Wraps model for gradient synchronization
- `train_loader.sampler.set_epoch(epoch)`: Ensures proper data shuffling across epochs
- `if rank == 0`: Only the first GPU prints to avoid duplicate logs

### 6. Process Spawning
```python
if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # How many GPUs?
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

**What this does:**
- `torch.cuda.device_count()`: Counts available GPUs
- `mp.spawn()`: Creates one process per GPU
- Each process runs `train_ddp()` with a different `rank`

## How DDP Works Behind the Scenes

1. **Data Distribution**: Each GPU gets different batches of data
2. **Forward Pass**: Each GPU computes forward pass on its data
3. **Backward Pass**: Each GPU computes gradients
4. **Gradient Synchronization**: DDP automatically averages gradients across all GPUs
5. **Parameter Update**: All GPUs update their parameters with the averaged gradients

## Parameter-Level Gradient Synchronization (Deep Dive)

Let's understand exactly how DDP synchronizes gradients at the parameter level:

### Example: Single Parameter in Attention Layer

Consider a single weight in the `out_proj` layer:
```python
self.out_proj = nn.Linear(768, 768)  # Weight matrix: [768, 768]
# Focus on out_proj.weight[0, 0] (first row, first column)
```

### Step-by-Step Process

**Initial State:**
```
Parameter value: out_proj.weight[0, 0] = 0.1234

GPU 0 processes: "The cat sat on"
GPU 1 processes: "Dogs love to play"
```

**Forward Pass (Different on Each GPU):**
```
GPU 0: context_vec[0, 0] = 0.5  (from "cat" context)
GPU 1: context_vec[0, 0] = 0.8  (from "dog" context)
```

**Backward Pass - Gradient Calculation:**
```python
# Each GPU computes gradient for the SAME parameter
GPU 0: out_proj.weight[0, 0].grad = -0.02  # Wants to decrease weight
GPU 1: out_proj.weight[0, 0].grad = +0.03  # Wants to increase weight
```

**DDP Synchronization:**
```python
# DDP averages the gradients for this specific parameter
averaged_grad = (-0.02 + 0.03) / 2 = +0.005

# Both GPUs get the same averaged gradient
GPU 0: out_proj.weight[0, 0].grad = +0.005
GPU 1: out_proj.weight[0, 0].grad = +0.005
```

**Parameter Update:**
```python
# Both GPUs apply the same update
new_weight = 0.1234 - (learning_rate * 0.005)
           = 0.1234 - (0.0004 * 0.005)
           = 0.123398

# Result: Both GPUs have identical weights
GPU 0: out_proj.weight[0, 0] = 0.123398
GPU 1: out_proj.weight[0, 0] = 0.123398
```

### This Happens for EVERY Parameter

For a 768×768 attention matrix, this synchronization happens **589,824 times simultaneously**:

```
Parameter [0,0]: GPU0_grad=-0.02, GPU1_grad=+0.03 → avg=+0.005
Parameter [0,1]: GPU0_grad=+0.01, GPU1_grad=-0.04 → avg=-0.015
Parameter [0,2]: GPU0_grad=+0.05, GPU1_grad=+0.02 → avg=+0.035
...
Parameter [767,767]: GPU0_grad=-0.01, GPU1_grad=+0.01 → avg=0.000
```

### Why This Is Powerful

**Different Data → Different Gradients → Better Combined Learning:**

```
GPU 0 learns: "cat" and "sat" should attend to each other
GPU 1 learns: "dogs" and "play" should attend to each other

Combined: Model learns BOTH patterns simultaneously!
```

### Visual Representation

```
Single Parameter: out_proj.weight[i, j]

GPU 0: data_batch_0 → forward → loss_0 → ∂loss/∂weight[i,j] = grad_0
GPU 1: data_batch_1 → forward → loss_1 → ∂loss/∂weight[i,j] = grad_1

DDP: final_grad = (grad_0 + grad_1) / 2

Both GPUs: weight[i,j] = weight[i,j] - lr * final_grad
```

### Key Insight

Each parameter learns from **ALL the data** across all GPUs, not just the data on its own GPU. The averaging ensures that every parameter benefits from the diverse gradients computed from different data batches.

**Result:** The model becomes more robust because each parameter has been updated based on a wider variety of training examples!

## Benefits of DDP

- **Speed**: Train faster by using multiple GPUs
- **Memory**: Distribute model and data across multiple GPUs
- **Scalability**: Easy to scale from 2 to 8+ GPUs
- **Better Learning**: Each parameter learns from diverse data across all GPUs
- **Robustness**: Model sees more varied examples per parameter update

## Running the Code

Simply run:
```bash
python3 gpt2.py
```

The script automatically detects your GPUs and uses all of them!

## Common Issues for Beginners

1. **CUDA Out of Memory**: Reduce batch size if you get memory errors
2. **Hanging**: Make sure all GPUs are available and not being used by other processes
3. **Different Results**: DDP training might give slightly different results due to different data ordering

## Summary

The main changes were:
1. Added distributed imports
2. Created setup/cleanup functions for GPU communication
3. Used DistributedSampler for data distribution
4. Wrapped model with DDP for gradient synchronization
5. Used multiprocessing to spawn one process per GPU

This transforms single-GPU training into efficient multi-GPU training!