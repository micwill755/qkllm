# PyTorch CNN Implementation

This is the PyTorch version of the NumPy CNN implementation.

## Key Differences from NumPy Version

### 1. **Model Definition**

**NumPy:**
```python
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2D(...)
        self.layers = [self.conv1, self.relu1, ...]
```

**PyTorch:**
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(...)
```

### 2. **Forward Pass**

**NumPy:**
```python
def forward(self, x):
    for layer in self.layers:
        x = layer.forward(x)
    return x
```

**PyTorch:**
```python
def forward(self, x):
    x = F.relu(self.conv1(x))
    return x
```

### 3. **Backward Pass**

**NumPy:**
```python
# Manual backward implementation
def backward(self, grad_output):
    grad_weights = np.zeros_like(self.weights)
    # ... compute gradients manually
    self.grad_weights = grad_weights
```

**PyTorch:**
```python
# Automatic!
loss.backward()  # Computes all gradients automatically
```

### 4. **Training Loop**

**NumPy:**
```python
# Manual gradient computation and updates
predictions = model.forward(batch_X)
loss, grad_loss = cross_entropy_loss(predictions, targets)
model.backward(grad_loss)
optimizer.step(model)
```

**PyTorch:**
```python
# Automatic differentiation
optimizer.zero_grad()
outputs = model(batch_X)
loss = criterion(outputs, batch_y)
loss.backward()  # Auto-computes gradients
optimizer.step()  # Auto-updates weights
```

### 5. **No Manual Gradient Implementation**

PyTorch handles all the backward pass math automatically using autograd!

## Running the Code

### Single GPU/CPU Training
```bash
cd qkllm/model/python/model/image/cnn/pytorch
python train_pytorch.py
```

### Multi-GPU Training with FSDP
```bash
cd qkllm/model/python/model/image/cnn/pytorch

# Single GPU (FSDP compatible)
python train_pytorch_fsdp.py

# Multi-GPU (e.g., 2 GPUs)
torchrun --nproc_per_node=2 train_pytorch_fsdp.py

# Or use the provided script
chmod +x run_fsdp.sh
./run_fsdp.sh
```

## Requirements

```bash
pip install torch torchvision
```

## Advantages of PyTorch

1. **Automatic differentiation** - No manual backward pass
2. **GPU acceleration** - Automatic CUDA support
3. **Built-in layers** - No need to implement Conv2D, MaxPool, etc.
4. **Optimized** - Highly optimized C++/CUDA backend
5. **Less code** - Much simpler and cleaner

## Comparison

| Feature | NumPy | PyTorch |
|---------|-------|---------|
| Lines of code | ~400 | ~80 |
| Backward pass | Manual | Automatic |
| GPU support | No | Yes |
| Speed | Slow | Fast |
| Learning value | High | Medium |
| Production ready | No | Yes |

The NumPy version is great for learning how CNNs work under the hood.
The PyTorch version is what you'd use in practice!

## FSDP (Fully Sharded Data Parallel)

FSDP is PyTorch's solution for efficient distributed training:

### What FSDP Does:
- **Shards model parameters** across GPUs (reduces memory per GPU)
- **Shards gradients** during backward pass
- **Shards optimizer states** (Adam, SGD, etc.)
- Enables training larger models than fit on a single GPU

### Key Features:
1. **Full Shard Strategy**: Shards everything (parameters, gradients, optimizer states)
2. **Auto Wrap Policy**: Automatically wraps layers based on size
3. **CPU Offloading**: Can offload to CPU for even larger models
4. **Mixed Precision**: Works with automatic mixed precision (AMP)

### When to Use FSDP:
- Training on multiple GPUs
- Model doesn't fit on single GPU
- Want to scale to larger batch sizes
- Need efficient distributed training

### Comparison:

| Method | Memory per GPU | Speed | Use Case |
|--------|---------------|-------|----------|
| Single GPU | Full model | Baseline | Small models |
| DataParallel (DP) | Full model | 1.5-2x | Legacy multi-GPU |
| DistributedDataParallel (DDP) | Full model | 3-4x | Standard multi-GPU |
| FSDP | Sharded model | 3-4x | Large models, memory constrained |

### Example Output:
```
Using device: cuda:0
World size: 2
Generating dummy data...
Creating model...
Wrapping model with FSDP...
Training...
Epoch 1/3, Loss: 2.3012, Accuracy: 10.50%
Epoch 2/3, Loss: 2.2845, Accuracy: 12.00%
Epoch 3/3, Loss: 2.2701, Accuracy: 14.50%

Evaluating...
Test Accuracy: 16.00%

Model saved to simple_cnn_fsdp.pth
```
