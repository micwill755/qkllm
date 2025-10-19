# CUDA GPT-2 Implementation

This directory contains a CUDA-accelerated implementation of the GPT-2 model, converted from the C implementation for GPU execution.

## Features

- **GPU-accelerated inference**: All major operations run on GPU
- **cuBLAS integration**: Optimized matrix multiplications
- **Memory efficient**: Minimal host-device transfers
- **Causal attention**: Proper autoregressive generation
- **Modular design**: Separate kernels for different operations

## Files

- `cuda_kernels.cu/cuh`: Core CUDA kernels (embedding, softmax, masking)
- `cuda_linear.cu/cuh`: GPU linear layers with cuBLAS
- `cuda_attention.cu/cuh`: Multi-head attention implementation
- `cuda_model.cu/cuh`: Complete GPT model
- `main.cu`: Example usage and testing
- `Makefile`: Build configuration

## Requirements

- NVIDIA GPU with compute capability 7.0+
- CUDA Toolkit 11.0+
- cuBLAS, cuRAND, cuDNN libraries

## Building

```bash
# Install dependencies (Ubuntu/Debian)
make install-deps

# Build
make

# Run
make run
```

## Architecture

The CUDA implementation mirrors the C version but with:

1. **Device memory management**: All weights and activations on GPU
2. **Kernel fusion**: Combined operations where possible
3. **Optimized attention**: Efficient causal masking and softmax
4. **Batch processing**: Ready for batch inference

## Performance

Expected speedups over CPU implementation:
- Linear layers: 10-50x (depending on size)
- Attention: 5-20x (memory bandwidth bound)
- Overall inference: 5-15x

## Usage

```cpp
CudaGPTConfig config = {
    .vocab_size = 50257,
    .context_length = 1024,
    .emb_dim = 768,
    .n_heads = 12,
    .n_layers = 12
};

CudaGPTModel model;
cuda_gpt_init(&model, &config);

int next_token = cuda_gpt_generate_token(&model, input_ids, seq_len);
```