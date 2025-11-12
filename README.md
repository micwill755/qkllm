# LLM

A collection of LLM implementations built from scratch using [QMX](https://pypi.org/project/qmx/) for accelerated matrix operations.

## Features

- **Pure Python implementations** of popular LLM architectures
- **Educational focus** - understand how LLMs work under the hood
- **QMX acceleration** - 274-417x faster than pure Python
- **No heavy dependencies** - built from scratch

## Implemented Models

### Llama 2
- RMSNorm layer normalization
- Multi-head attention with RoPE
- Feed-forward networks
- Grouped Query Attention (GQA)

**Location**: `model/python/model/llama/`

## Installation

```bash
cd llm
pip install -r requirements.txt
```

This will install:
- `qmx` - Fast matrix operations
- `tiktoken` - Tokenization

## Usage

### Llama 2 Example

```python
from model.python.model.llama.llama2.llama2 import Llama2Model, LLAMA2_CONFIG_MINI
import mx

# Create model
model = Llama2Model(LLAMA2_CONFIG_MINI)

# Create input
tokens = [1, 2, 3, 4, 5]
input = mx.Tensor([tokens])

# Forward pass
output = model(input)
print(output.shape)  # (batch, seq_len, emb_dim)
```

### Run Tests

```bash
cd model/python/model/llama/llama2
python llama2.py
```

## Project Structure

```
llm/
├── model/
│   ├── python/
│   │   └── model/
│   │       └── llama/
│   │           ├── attention.py       # Multi-head attention
│   │           ├── feed_forward.py    # Feed-forward layers
│   │           ├── rope.py            # Rotary Position Embedding
│   │           └── llama2/
│   │               ├── llama2.py      # Llama 2 model
│   │               └── embedding.py   # Token embeddings
│   ├── c/                             # C implementations
│   └── cuda/                          # CUDA implementations
├── inference/                         # Inference optimizations
├── training/                          # Training utilities
└── requirements.txt
```

## Components

### Attention Mechanisms
- **Multi-Head Attention** - Standard transformer attention
- **Grouped Query Attention** - Efficient attention for Llama 2
- **RoPE** - Rotary Position Embeddings

### Normalization
- **RMSNorm** - Root Mean Square Layer Normalization (Llama 2)
- **LayerNorm** - Standard layer normalization

### Layers
- **Linear** - Fully connected layers
- **Embedding** - Token embeddings
- **Feed-Forward** - MLP blocks

## Performance

Using QMX for matrix operations:

| Operation | Pure Python | QMX (C) | Speedup |
|-----------|-------------|---------|---------|
| 64x64 matmul | 90.72 ms | 0.22 ms | **417x** |
| 128x128 matmul | 575.99 ms | 1.82 ms | **316x** |
| 256x256 matmul | 4550.17 ms | 16.58 ms | **274x** |

## Configuration

### Llama 2 Mini (for testing)
```python
LLAMA2_CONFIG_MINI = {
    "vocab_size": 100,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 512
}
```

### Llama 2 7B (full model)
```python
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,
    "context_length": 4096,
    "emb_dim": 4096,
    "n_heads": 32,
    "n_layers": 32,
    "hidden_dim": 11008
}
```

## Dependencies

- **qmx** - Fast matrix operations ([PyPI](https://pypi.org/project/qmx/))
- **tiktoken** - Tokenization

## Roadmap

- [ ] GPT-2 implementation
- [ ] Training loop
- [ ] Inference optimizations
- [ ] Model quantization
- [ ] Distributed training
- [ ] CUDA kernels

## Contributing

Contributions welcome! This is an educational project focused on understanding LLMs from first principles.

## License

MIT License

## Links

- **QMX Package**: https://pypi.org/project/qmx/
- **GitHub**: https://github.com/yourusername/llm
