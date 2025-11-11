# GPU Collective Communication Operations

A comprehensive guide to collective operations used in distributed deep learning.

---

## **1. Broadcast**

### What it does
One GPU (root) sends its data to all other GPUs.

### Important Note
**For large models (Tensor/Pipeline Parallelism):** Broadcast is NOT used for model weights. Instead:
- Each GPU loads only its shard directly from disk
- Avoids materializing full model on any single GPU
- Example: GPT-3 (175B params) - each GPU loads ~25GB shard, never the full 350GB model

### Example
```
Before:
GPU 0: [1, 2, 3]  ← root/source
GPU 1: [?, ?, ?]
GPU 2: [?, ?, ?]

After:
GPU 0: [1, 2, 3]
GPU 1: [1, 2, 3]  ← copied from GPU 0
GPU 2: [1, 2, 3]  ← copied from GPU 0
```

### Why we use it
- **Model initialization (Data Parallelism only)**: Broadcast initial weights from rank 0 to all GPUs when model fits in memory
- **Hyperparameters**: Share learning rate, batch size across all workers (always small)
- **Random seeds**: Ensure all GPUs use same random state for reproducibility
- **Control signals**: Send stop/continue signals to all workers

### Communication cost
`O(N)` where N = number of GPUs (using tree-based broadcast)

---

## **2. Reduce**

### What it does
All GPUs send data to one root GPU, which performs a reduction operation (sum, max, min, etc.).

### Important Note
**Reduce is for SMALL tensors only (even with large models):**
- ✅ Loss values (scalar)
- ✅ Accuracy metrics (scalar)
- ✅ Gradient norms (scalar)
- ❌ NOT for gradients (use All-Reduce - all GPUs need them)
- ❌ NOT for model weights (never reduced in Tensor/Pipeline Parallelism)

### Example
```
Before:
GPU 0: [1, 2, 3]
GPU 1: [4, 5, 6]
GPU 2: [7, 8, 9]

After (SUM to GPU 0):
GPU 0: [12, 15, 18]  ← sum of all
GPU 1: [4, 5, 6]     ← unchanged
GPU 2: [7, 8, 9]     ← unchanged
```

### Why we use it
- **Metrics aggregation**: Collect loss/accuracy from all GPUs to rank 0 for logging (ALWAYS small tensors)
- **Validation results**: Sum correct predictions across GPUs for accuracy calculation
- **Monitoring**: Aggregate training statistics (gradient norms, learning rate, etc.)

### Example with Large Models
```python
# Training step on GPT-3 with Tensor Parallelism
# Each GPU has 25GB shard

# Forward + Backward (uses All-Reduce for activations/gradients)
output = model(input)
loss = criterion(output, target)
loss.backward()

# REDUCE loss to GPU 0 for logging (tiny scalar!)
if rank == 0:
    total_loss = dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
    print(f"Loss: {total_loss / world_size}")

# Optimizer step (each GPU updates its own shard)
optimizer.step()
```

### Communication cost
`O(N)` - all GPUs send to root, but only for small tensors

---

## **3. All-Reduce** 

### What it does
Reduce + Broadcast in one operation. Every GPU gets the reduced result.

### Example
```
Before:
GPU 0: [1, 2, 3]
GPU 1: [4, 5, 6]
GPU 2: [7, 8, 9]

After (SUM):
GPU 0: [12, 15, 18]  ← sum on all GPUs
GPU 1: [12, 15, 18]  ← sum on all GPUs
GPU 2: [12, 15, 18]  ← sum on all GPUs
```

### Why we use it
- **Data Parallelism - Gradient sync**: Synchronize gradients across all GPUs after backward pass (CRITICAL!)
- **Tensor Parallelism - Row-parallel layers**: Sum partial outputs from tensor-parallel GPUs (your line 109!)
- **Distributed optimizer**: Average optimizer states across workers
- **Most common operation** in distributed training

### Reduce vs All-Reduce
**Key difference:** Who needs the result?

**Reduce:** Only one GPU (rank 0) needs result
- Use case: Logging metrics to console/file
- Example: `loss = 0.5` → only rank 0 prints it

**All-Reduce:** ALL GPUs need the result
- Use case: Gradient synchronization (all GPUs need gradients to update weights)
- Example: Each GPU computed partial gradient → all need full gradient

### Communication cost
`O(N)` using Ring All-Reduce algorithm (bandwidth optimal)

### Ring All-Reduce Algorithm
```
Step 1: Scatter-Reduce (each GPU reduces chunks)
Step 2: All-Gather (each GPU gathers all chunks)
Result: All GPUs have full reduced tensor
```

---

## **4. Gather**

### What it does
One root GPU collects (concatenates) data from all GPUs.

### Example
```
Before:
GPU 0: [1, 2]
GPU 1: [3, 4]
GPU 2: [5, 6]

After (Gather to GPU 0):
GPU 0: [1, 2, 3, 4, 5, 6]  ← concatenated
GPU 1: [3, 4]              ← unchanged
GPU 2: [5, 6]              ← unchanged
```

### Why we use it
- **Batch assembly**: Collect mini-batches from all GPUs for evaluation
- **Predictions**: Gather model outputs to one GPU for saving/analysis
- **Debugging**: Collect intermediate activations to one GPU for inspection

### Communication cost
`O(N)` - all GPUs send to root

---

## **5. All-Gather** ⭐

### What it does
Gather + Broadcast. Every GPU gets concatenated data from all GPUs.

### Example
```
Before:
GPU 0: [1, 2]
GPU 1: [3, 4]
GPU 2: [5, 6]

After:
GPU 0: [1, 2, 3, 4, 5, 6]  ← all data
GPU 1: [1, 2, 3, 4, 5, 6]  ← all data
GPU 2: [1, 2, 3, 4, 5, 6]  ← all data
```

### Why we use it
- **Column-parallel layers**: Reconstruct full tensor from sharded outputs
- **Tensor Parallelism backward**: Gather gradients split across GPUs
- **Sequence parallelism**: Combine sequence chunks from different GPUs
- **Embedding tables**: Gather embedding shards for full vocabulary

### Communication cost
`O(N)` using ring-based algorithm

---

## **6. Scatter**

### What it does
One root GPU splits its data and sends different chunks to each GPU.

### Example
```
Before:
GPU 0: [1, 2, 3, 4, 5, 6]  ← root
GPU 1: [?, ?]
GPU 2: [?, ?]

After:
GPU 0: [1, 2]
GPU 1: [3, 4]
GPU 2: [5, 6]
```

### Why we use it
- **Data distribution**: Split batch from rank 0 to all workers
- **Work distribution**: Divide tasks across GPUs
- **Load balancing**: Distribute uneven workloads

### Communication cost
`O(N)` - root sends to all

---

## **7. Reduce-Scatter**

### What it does
Reduce + Scatter. Performs reduction then splits result across GPUs.

### Example
```
Before:
GPU 0: [1, 2, 3, 4]
GPU 1: [5, 6, 7, 8]

After (SUM):
GPU 0: [6, 8]      ← sum of first half [1+5, 2+6]
GPU 1: [10, 12]    ← sum of second half [3+7, 4+8]
```

### Why we use it
- **Optimized gradient sync**: More efficient than all-reduce when only partial results needed
- **ZeRO optimizer**: Partition optimizer states across GPUs
- **Memory optimization**: Reduce memory by keeping only needed chunks
- **Building block**: Used internally in Ring All-Reduce

### Communication cost
`O(N)` - more efficient than separate reduce + scatter

---

## **Model Loading Strategies by Parallelism Type**

### Data Parallelism (Small-Medium Models)
```python
# Step 1: Load full model on GPU 0
if rank == 0:
    model = MyModel()  # 1GB model
    model.load_state_dict(torch.load('checkpoint.pt'))
else:
    model = MyModel()  # Uninitialized

# Step 2: BROADCAST weights to all GPUs
for param in model.parameters():
    dist.broadcast(param.data, src=0)

# Result: All GPUs have identical full model
GPU 0: [Full Model - 1GB]
GPU 1: [Full Model - 1GB]  ← Broadcasted
GPU 2: [Full Model - 1GB]  ← Broadcasted
```
**Use case:** ResNet, BERT-base, models that fit in single GPU memory

---

### Tensor Parallelism (Large Models)
```python
# NO broadcast! Each GPU loads only its shard
rank = dist.get_rank()
world_size = dist.get_world_size()

# Step 1: Load checkpoint metadata (small)
checkpoint_meta = torch.load('checkpoint_meta.pt')

# Step 2: Each GPU loads ONLY its shard from disk
shard_path = f'checkpoint_shard_{rank}.pt'
model_shard = torch.load(shard_path, map_location=f'cuda:{rank}')

# Result: Each GPU has different shard, never full model
GPU 0: [Shard 0 - 25GB]  ← Loaded directly from disk
GPU 1: [Shard 1 - 25GB]  ← Loaded directly from disk
GPU 2: [Shard 2 - 25GB]  ← Loaded directly from disk
GPU 3: [Shard 3 - 25GB]  ← Loaded directly from disk

# Full model (100GB) never materialized on any GPU!
```
**Use case:** GPT-3 (175B), LLaMA (70B), models too large for single GPU

---

### Pipeline Parallelism (Large Models)
```python
# Each GPU loads only its assigned layers
rank = dist.get_rank()
layers_per_gpu = total_layers // world_size

# Step 1: Determine which layers this GPU owns
start_layer = rank * layers_per_gpu
end_layer = (rank + 1) * layers_per_gpu

# Step 2: Load only those layers from checkpoint
for layer_idx in range(start_layer, end_layer):
    layer_weights = torch.load(f'layer_{layer_idx}.pt')
    model.layers[layer_idx].load_state_dict(layer_weights)

# Result: Each GPU has different layers
GPU 0: [Layers 0-10]   ← 25GB
GPU 1: [Layers 11-20]  ← 25GB
GPU 2: [Layers 21-30]  ← 25GB
GPU 3: [Layers 31-40]  ← 25GB
```
**Use case:** Very deep models, memory-constrained scenarios

---

## **Usage in Tensor Parallel MLP**

### Initialization (One-time)
```python
# NO broadcast of model weights!
# Each GPU loads its shard directly:
rank = dist.get_rank()
hidden_per_shard = C_hidden // tp_size
start = rank * hidden_per_shard
end = (rank + 1) * hidden_per_shard

# Load only this GPU's shard
W1_shard = full_weights[:, start:end]  # Column-parallel
W2_shard = full_weights[start:end, :]  # Row-parallel
```

### Forward Pass
```python
# Column-parallel fc1: No communication needed
# Each GPU computes: h[i] = x @ W1[i]

# Row-parallel fc2: ALL-REDUCE needed
y = all_reduce_sum([h[0] @ W2[0], h[1] @ W2[1], ...])
```

### Backward Pass
```python
# Row-parallel fc2 backward: No communication (gradients already split)

# Column-parallel fc1 backward: ALL-GATHER needed
# Reconstruct full gradient from sharded pieces
```

---

## **Communication Libraries**

- **NCCL** (NVIDIA): Optimized for NVIDIA GPUs, used by PyTorch
- **Gloo**: CPU and GPU, used by PyTorch for CPU operations
- **MPI**: Traditional HPC, works across different hardware
- **RCCL** (AMD): AMD's version of NCCL

---

## **Performance Tips**

1. **Overlap communication with computation**: Start all-reduce while computing next layer
2. **Gradient bucketing**: Group small tensors to reduce communication overhead
3. **Compression**: Use FP16 or gradient compression to reduce bandwidth
4. **Topology-aware**: Use operations optimized for your network topology (NVLink, InfiniBand)

---

## **Summary Table**

| Operation | Input Distribution | Output Distribution | Primary Use Case |
|-----------|-------------------|---------------------|------------------|
| Broadcast | One GPU | All GPUs (same) | Model initialization |
| Reduce | All GPUs | One GPU | Metrics logging |
| All-Reduce | All GPUs | All GPUs (same) | Gradient sync (DP) |
| Gather | All GPUs | One GPU (concat) | Collect predictions |
| All-Gather | All GPUs | All GPUs (concat) | Tensor parallel backward |
| Scatter | One GPU | All GPUs (split) | Data distribution |
| Reduce-Scatter | All GPUs | All GPUs (split) | ZeRO optimizer |
