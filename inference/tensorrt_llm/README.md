# TensorRT-LLM Beginner's Guide

## What is TensorRT-LLM?

TensorRT-LLM is NVIDIA's optimized inference engine for large language models. It makes your AI models run **2-5x faster** by pre-compiling them into optimized "engines."

## Key Concept: What is an Engine?

A **TensorRT engine** is a pre-compiled, hardware-optimized version of your neural network model.

### PyTorch Model (Original)
```python
model = AutoModelForCausalLM.from_pretrained("model_name")
# - Contains model weights and architecture
# - Uses generic PyTorch operations (nn.Linear, nn.LayerNorm, etc.)
# - Operations are interpreted at runtime
# - Works on any compatible hardware
# - Slower execution due to runtime overhead
```

### TensorRT Engine (Optimized)
```python
engine = build_tensorrt_engine(model, target_gpu="A100")
# - Same model weights, optimized architecture
# - Uses fused CUDA kernels specific to your GPU
# - Operations are pre-compiled for your hardware
# - Only works on the target GPU architecture
# - 2-5x faster execution, lower memory usage
```

### What Happens During Engine Building?

1. **Operation Fusion**: Multiple operations combined into single GPU kernels
   ```python
   # Before: 3 separate operations
   x = layer_norm(x)
   x = linear_layer(x) 
   x = activation(x)
   
   # After: 1 fused kernel
   x = fused_ln_linear_activation(x)  # Much faster
   ```

2. **Memory Layout Optimization**: Data arranged for optimal GPU memory access patterns

3. **Precision Optimization**: Uses FP16 or INT8 where possible without accuracy loss

4. **Hardware-Specific Kernels**: Generated for your exact GPU model (RTX 4090, A100, H100, etc.)

5. **Graph Optimization**: Removes unnecessary operations, optimizes computation order

## Step-by-Step Tutorial

### Step 1: Install Dependencies
```bash
pip install tensorrt-llm transformers torch
```

### Step 2: Build the Engine (One-time Setup)
```python
# build_engine.py - Run this ONCE
python build_engine.py
```
This creates an optimized engine file from the original model.

**What happens during engine building:**
- Downloads TinyLlama model weights from HuggingFace
- Analyzes the neural network graph (layers, connections, operations)
- Applies hardware-specific optimizations for your GPU
- Compiles optimized CUDA kernels
- Saves the compiled engine to `tinyllama_engine/` folder
- **Note**: This process can take 10-30 minutes depending on model size

### Step 3: Use the Engine for Inference
```python
# basic.py - Run this for actual inference
python basic.py
```

## Code Walkthrough: basic.py

### 1. Import Libraries
```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, SamplingConfig
from transformers import AutoTokenizer
```

### 2. Define Your Prompts
```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
]
```

### 3. Load Tokenizer
```python
# TensorRT-LLM needs separate tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### 4. Configure Generation Settings
```python
sampling_config = SamplingConfig(
    temperature=0.8,    # Creativity level (0.0 = deterministic, 1.0 = creative)
    top_p=0.95,        # Focus on top 95% probable tokens
    max_new_tokens=50  # Maximum tokens to generate
)
```

### 5. Load the Pre-built Engine
```python
runner = ModelRunner.from_dir(
    engine_dir="tinyllama_engine",  # Path to your built engine
    lora_dir=None,
    rank=0
)
```

### 6. Tokenize Input
```python
batch_input_ids = []
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    batch_input_ids.append(input_ids)
```

### 7. Generate Text
```python
outputs = runner.generate(
    batch_input_ids=batch_input_ids,
    sampling_config=sampling_config
)
```

### 8. Decode and Display Results
```python
for i, output in enumerate(outputs):
    prompt = prompts[i]
    # Skip input tokens, only decode generated tokens
    generated_ids = output[0][len(batch_input_ids[i][0]):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Comparison: vLLM vs TensorRT-LLM

| Aspect | vLLM | TensorRT-LLM |
|--------|------|--------------|
| **Setup** | Simple - direct model loading | Complex - requires engine building |
| **Speed** | Good | Excellent (2-5x faster) |
| **Memory** | Higher usage | Lower usage |
| **Flexibility** | High - change models easily | Low - engines are model-specific |
| **Best for** | Development, experimentation | Production, high-throughput |

### vLLM Code (Simple)
```python
from vllm import LLM, SamplingParams

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # One line!
outputs = llm.generate(prompts, sampling_params)
```

### TensorRT-LLM Code (More Setup, Better Performance)
```python
# 1. Build engine (one-time)
# 2. Load tokenizer
# 3. Configure sampling
# 4. Load engine
# 5. Tokenize manually
# 6. Generate
# 7. Decode manually
```

## When to Use TensorRT-LLM

**Use TensorRT-LLM when:**
- ✅ You need maximum performance
- ✅ Running in production with high traffic
- ✅ Using NVIDIA GPUs
- ✅ Model architecture is stable

**Use vLLM when:**
- ✅ Rapid prototyping
- ✅ Experimenting with different models
- ✅ Simple setup preferred
- ✅ Development phase

## Common Issues & Solutions

### Issue: "Engine not found"
**Solution:** Run `build_engine.py` first to create the engine.

### Issue: "CUDA out of memory"
**Solution:** Reduce `max_batch_size` in engine building or use smaller models.

### Issue: "Tokenizer mismatch"
**Solution:** Ensure tokenizer model name matches the engine's original model.

## Next Steps

1. **Try the basic example** - Run through the tutorial
2. **Experiment with parameters** - Change temperature, top_p
3. **Try different models** - Build engines for other models
4. **Optimize for your use case** - Adjust batch sizes, sequence lengths
5. **Deploy to production** - Use the optimized engines in your applications

## Performance Tips

- **Batch requests** when possible for better throughput
- **Use appropriate precision** (FP16 vs FP32) based on your needs
- **Profile your workload** to find optimal engine settings
- **Keep engines warm** - first inference is slower due to GPU initialization