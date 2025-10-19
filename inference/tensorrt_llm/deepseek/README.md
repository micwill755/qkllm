# DeepSeek V3 TensorRT-LLM Implementation

This directory contains the TensorRT-LLM compatible implementation of DeepSeek V3 model with Multi-Head Latent Attention and Mixture of Experts.

## Files Structure

- `configuration_deepseek.py` - Configuration class for DeepSeek V3
- `modeling_deepseek.py` - TensorRT-LLM compatible model implementation
- `build_deepseek.py` - Build script for creating optimized models
- `example_usage.py` - Example usage and testing script
- `deepseek_v3_pytorch_tensorrt_llm.py` - Original PyTorch implementation with TensorRT-LLM integration

## Key Features

- **Multi-Head Latent Attention**: Efficient attention mechanism using latent tokens
- **Mixture of Experts (MoE)**: Sparse expert routing for scalability
- **TensorRT-LLM Optimizations**: 
  - Tensor parallelism support
  - Quantization ready
  - Optimized kernels for attention and linear layers
  - Memory-efficient inference

## Quick Start

### 1. Build the Model

```bash
python build_deepseek.py \
    --model_dir /path/to/deepseek/checkpoint \
    --output_dir ./deepseek_tensorrt \
    --tp_size 1 \
    --dtype float16 \
    --max_batch_size 8 \
    --max_input_len 1024 \
    --max_output_len 1024
```

### 2. Test the Implementation

```bash
python example_usage.py
```

### 3. Use in Your Code

```python
from tensorrt_llm._torch.runtime import TorchRuntime
from deepseek import DeepSeekV3ForCausalLM, DeepSeekV3Config

# Load built model
runtime = TorchRuntime.load("./deepseek_tensorrt")

# Generate text
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
outputs = runtime.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9
)
```

## Model Architecture

### Multi-Head Latent Attention
- Compresses input sequence to latent tokens
- Cross-attention between input and latent representations
- Reduces computational complexity from O(n²) to O(n×k) where k << n

### Mixture of Experts
- Router network selects top-k experts per token
- Sparse computation for better scaling
- Load balancing to prevent expert collapse

### Optimizations
- RMSNorm for stable training
- RoPE (Rotary Position Embedding)
- SwiGLU activation in experts
- Tensor parallelism support

## Performance Benefits

TensorRT-LLM provides several optimizations:

1. **Kernel Fusion**: Fused attention and MLP kernels
2. **Memory Optimization**: KV-cache management and paged attention
3. **Quantization**: INT8/FP16 support for faster inference
4. **Parallelism**: Multi-GPU tensor and pipeline parallelism
5. **Dynamic Batching**: Efficient batch processing

## Configuration Options

Key configuration parameters:

- `hidden_size`: Model dimension (768, 1024, 2048, etc.)
- `num_hidden_layers`: Number of transformer layers
- `num_attention_heads`: Number of attention heads
- `num_experts`: Total number of experts in MoE
- `num_experts_per_tok`: Top-k experts selected per token
- `latent_dim`: Dimension of latent attention space
- `max_position_embeddings`: Maximum sequence length

## Requirements

- TensorRT-LLM >= 0.7.0
- PyTorch >= 2.0
- CUDA >= 11.8
- Python >= 3.8

## Troubleshooting

1. **Import Errors**: Ensure TensorRT-LLM is properly installed
2. **CUDA OOM**: Reduce batch size or sequence length
3. **Slow Performance**: Enable tensor parallelism for large models
4. **Accuracy Issues**: Check weight loading and data types

## Next Steps

1. Add support for different quantization modes
2. Implement custom CUDA kernels for latent attention
3. Add pipeline parallelism for very large models
4. Optimize expert routing for better load balancing