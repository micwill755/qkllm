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

```bash
cd qkllm/model/python/model/image/cnn/pytorch
python train_pytorch.py
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
