# mtrx

A minimal tensor computation library built from scratch in Python.

## Features

- **Tensor operations**: Create, manipulate, and compute with multi-dimensional tensors
- **Matrix operations**: Matrix multiplication, transpose, reshape
- **Neural network primitives**: Linear layers, softmax, masking
- **Initialization utilities**: Random, zeros, ones

## Installation

Install in editable mode from the project root:

```bash
cd /path/to/llm
pip install -e .
```

## Usage

```python
from mtrx.mtrx import ones, zeros, randn, mat_mul, softmax
from mtrx.tensor import Tensor
from mtrx.linear import Linear

# Create tensors
x = randn((2, 3, 4))
y = ones((4, 5))
z = zeros((3, 3))

# Matrix multiplication
result = mat_mul(x, y)

# Linear layer
layer = Linear(d_in=128, d_out=256)
output = layer(x)

# Softmax
probs = softmax(x)
```

## API

### Tensor Creation
- `randn(shape)` - Random tensor with Gaussian distribution
- `zeros(shape)` - Tensor filled with zeros
- `ones(shape)` - Tensor filled with ones

### Operations
- `mat_mul(m1, m2)` - Matrix multiplication
- `softmax(tensor)` - Softmax activation
- `mask(m, window)` - Apply causal masking
- `reshape(m, shape)` - Reshape tensor

### Modules
- `Linear(d_in, d_out)` - Linear transformation layer

## Requirements

- Python >= 3.7
