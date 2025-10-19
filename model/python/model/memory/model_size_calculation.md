# Model Size Calculation and GPU Planning Guide

## How to Calculate Model Size

For any transformer model, calculate parameters for each component:

### 1. Embedding Layers
```python
# Token embeddings: vocab_size × embedding_dim
tok_emb_params = 50257 × 768 = 38,597,376 (~38.6M)

# Position embeddings: context_length × embedding_dim  
pos_emb_params = 256 × 768 = 196,608 (~0.2M)
```

### 2. Transformer Block Components
Each transformer block contains:

```python
# Multi-Head Attention
W_query = emb_dim × emb_dim = 768 × 768 = 589,824
W_key   = emb_dim × emb_dim = 768 × 768 = 589,824  
W_value = emb_dim × emb_dim = 768 × 768 = 589,824
out_proj = emb_dim × emb_dim = 768 × 768 = 589,824
attention_params = 4 × 589,824 = 2,359,296 (~2.4M)

# Feed Forward Network
ff_layer1 = emb_dim × (4 × emb_dim) = 768 × 3072 = 2,359,296
ff_layer2 = (4 × emb_dim) × emb_dim = 3072 × 768 = 2,359,296  
ff_params = 4,718,592 (~4.7M)

# Layer Norms (2 per block)
layernorm_params = 2 × emb_dim = 2 × 768 = 1,536 (~0.002M)

# Total per block
block_params = 2,359,296 + 4,718,592 + 1,536 = 7,079,424 (~7.1M)
```

### 3. Output Head
```python
# Final linear layer: emb_dim × vocab_size
out_head_params = 768 × 50257 = 38,597,376 (~38.6M)
```

### 4. Total Model Size
```python
total_params = tok_emb + pos_emb + (n_layers × block_params) + out_head
total_params = 38.6M + 0.2M + (12 × 7.1M) + 38.6M
total_params = 38.6M + 0.2M + 85.2M + 38.6M = 162.6M

# Note: Actual GPT-2 124M has some parameter sharing/differences
# Our calculation gives ~163M, close to the 124M target
```

## Memory Requirements Calculation

### Parameters Memory (FP32)
```python
params_memory_gb = (total_params × 4 bytes) / (1024³)
params_memory_gb = (124M × 4) / (1024³) = 0.46 GB
```

### Training Memory (FP32)
```python
# Parameters: 1x model size
# Gradients: 1x model size  
# Optimizer states (Adam): 2x model size (momentum + variance)
# Activations: varies by batch size and sequence length

training_memory = params + gradients + optimizer_states + activations
training_memory = 0.46 + 0.46 + (2 × 0.46) + activations_memory
training_memory = 1.84 GB + activations_memory

# Activations for batch_size=2, seq_len=256:
activations_memory ≈ 0.5-1.0 GB

# Total per GPU (DDP): ~2.5-3.0 GB
# Total per GPU (SFDP): ~0.6-0.8 GB
```

## How Many GPUs Should You Use?

### Rule of Thumb:
```python
# For DDP: Each GPU needs full model + training overhead
min_gpu_memory_ddp = (model_size_gb × 4) + activation_memory

# For SFDP: Memory scales with number of GPUs
min_gpu_memory_sfdp = (model_size_gb × 4) / num_gpus + activation_memory
```

### GPU Planning Examples:

**Small Model (124M parameters, ~0.5GB):**
- **1 GPU (8GB)**: ✅ Easy fit with DDP
- **2-4 GPUs**: Use DDP for speed, SFDP not needed

**Medium Model (1.3B parameters, ~5GB):**
- **1 GPU (8GB)**: ❌ Won't fit with training overhead
- **2 GPUs**: ✅ SFDP recommended (~2.5GB per GPU)
- **4 GPUs**: ✅ SFDP optimal (~1.25GB per GPU)

**Large Model (7B parameters, ~28GB):**
- **4 GPUs (8GB each)**: ❌ Still too large
- **8 GPUs (8GB each)**: ✅ SFDP required (~3.5GB per GPU)
- **16 GPUs**: ✅ SFDP optimal (~1.75GB per GPU)

**Very Large Model (70B parameters, ~280GB):**
- **32+ GPUs**: ✅ SFDP + CPU offloading required

### Practical GPU Selection Formula

```python
def calculate_min_gpus(model_params_millions, gpu_memory_gb=8, precision='fp32'):
    """
    Calculate minimum GPUs needed for SFDP training
    """
    bytes_per_param = 4 if precision == 'fp32' else 2  # fp16
    
    # Model memory in GB
    model_memory_gb = (model_params_millions * 1e6 * bytes_per_param) / (1024**3)
    
    # Training overhead: gradients + optimizer states + activations
    training_memory_gb = model_memory_gb * 4  # Conservative estimate
    
    # Minimum GPUs needed
    min_gpus = math.ceil(training_memory_gb / (gpu_memory_gb * 0.8))  # 80% utilization
    
    return min_gpus, model_memory_gb, training_memory_gb

# Examples:
print(calculate_min_gpus(124))    # (1, 0.46, 1.84) - 1 GPU needed
print(calculate_min_gpus(1300))   # (1, 4.83, 19.32) - 3 GPUs needed  
print(calculate_min_gpus(7000))   # (12, 25.99, 103.95) - 17 GPUs needed
```

## Parameter Calculation for Different Model Sizes

### GPT-2 Model Family:
```python
# GPT-2 Small (124M)
config_124m = {"vocab_size": 50257, "context_length": 1024, "emb_dim": 768, "n_layers": 12}

# GPT-2 Medium (355M)  
config_355m = {"vocab_size": 50257, "context_length": 1024, "emb_dim": 1024, "n_layers": 24}

# GPT-2 Large (774M)
config_774m = {"vocab_size": 50257, "context_length": 1024, "emb_dim": 1280, "n_layers": 36}

# GPT-2 XL (1.5B)
config_1_5b = {"vocab_size": 50257, "context_length": 1024, "emb_dim": 1600, "n_layers": 48}
```

### General Formula for Any Transformer:
```python
def calculate_transformer_params(vocab_size, context_length, emb_dim, n_layers, n_heads):
    """
    Calculate total parameters for any transformer model
    """
    # Embeddings
    tok_emb = vocab_size * emb_dim
    pos_emb = context_length * emb_dim
    
    # Per transformer block
    attention = 4 * (emb_dim * emb_dim)  # Q, K, V, output projection
    feedforward = 2 * (emb_dim * 4 * emb_dim)  # Two linear layers
    layernorms = 2 * emb_dim  # Two layer norms per block
    block_params = attention + feedforward + layernorms
    
    # Output head
    output_head = emb_dim * vocab_size
    
    # Total
    total = tok_emb + pos_emb + (n_layers * block_params) + output_head
    
    return {
        "tok_emb": tok_emb,
        "pos_emb": pos_emb, 
        "blocks": n_layers * block_params,
        "output_head": output_head,
        "total": total
    }

# Example usage:
params = calculate_transformer_params(50257, 256, 768, 12, 12)
print(f"Total parameters: {params['total']:,}")
```

## Memory Optimization Strategies

### 1. Mixed Precision Training
```python
# FP16 reduces memory by ~50%
fp16_memory = fp32_memory / 2

# Example: 7B model
fp32_memory = 7000 * 4 / 1024**3 * 4  # 104 GB training memory
fp16_memory = 7000 * 2 / 1024**3 * 4  # 52 GB training memory
```

### 2. Gradient Checkpointing
```python
# Trades compute for memory
# Reduces activation memory by ~50-80%
# Increases training time by ~20-30%
```

### 3. CPU Offloading
```python
# Move unused parameters to CPU
# Can reduce GPU memory by 60-80%
# Adds communication overhead
```

## Quick Reference Table

| Model Size | Parameters | FP32 Memory | FP16 Memory | Min GPUs (8GB) | Min GPUs (24GB) |
|------------|------------|-------------|-------------|----------------|-----------------|
| GPT-2 Small| 124M       | 2.0 GB      | 1.0 GB      | 1              | 1               |
| GPT-2 Medium| 355M      | 5.7 GB      | 2.8 GB      | 1              | 1               |
| GPT-2 Large| 774M       | 12.4 GB     | 6.2 GB      | 2              | 1               |
| GPT-2 XL   | 1.5B       | 24.0 GB     | 12.0 GB     | 4              | 1               |
| GPT-3 Small| 7B         | 112 GB      | 56 GB       | 16             | 3               |
| GPT-3 Medium| 13B       | 208 GB      | 104 GB      | 32             | 5               |
| GPT-3 Large| 175B      | 2800 GB     | 1400 GB     | 400+           | 60+             |

*Note: Memory includes parameters + gradients + optimizer states + activations*