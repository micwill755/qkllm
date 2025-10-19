# LoRA Adapter for GPT-2: Beginner's Guide

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that allows you to adapt large language models with minimal computational resources. Instead of updating all model parameters, LoRA adds small trainable matrices to existing layers.

## Key Benefits

- **Memory Efficient**: Only trains 0.1-1% of original parameters
- **Fast Training**: Significantly faster than full fine-tuning
- **Modular**: Can swap different LoRA adapters for different tasks
- **Storage Efficient**: LoRA weights are much smaller than full model weights

## How LoRA Works

LoRA decomposes weight updates into two low-rank matrices:
```
W_new = W_original + A × B
```
Where:
- `W_original`: Frozen pre-trained weights
- `A`: Trainable matrix (d × r)
- `B`: Trainable matrix (r × k)
- `r`: Rank (much smaller than d or k)

## Project Structure

```
lora_adapter/
├── README.md              # This guide
├── lora_linear.py         # LoRA-enabled linear layer
├── peft_wrapper.py        # PEFT-style wrapper
├── examples/             # Usage examples
│   ├── basic_usage.py    # Basic LoRA example
│   └── fine_tune.py      # Fine-tuning demonstration
└── requirements.txt      # Dependencies
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Basic Usage**:
   ```python
   from gpt2 import GPT2Model, GPT_CONFIG_124M
   from peft_wrapper import get_peft_model
   
   # Use your existing GPT-2 model
   base_model = GPT2Model(GPT_CONFIG_124M)
   
   # Wrap with LoRA adapters (like Hugging Face PEFT)
   model = get_peft_model(base_model, target_modules=["out_head"], lora_rank=16)
   
   # Enable LoRA training
   model.enable_lora_training()
   ```

3. **Run Examples**:
   ```bash
   python examples/basic_usage.py
   python examples/fine_tune.py
   ```

## Configuration

Key LoRA parameters:
- `lora_rank`: Controls adapter size (4-64 typical)
- `lora_alpha`: Scaling factor (usually 16-32)
- `lora_dropout`: Regularization (0.1 typical)

## Next Steps

1. Start with `examples/basic_usage.py`
2. Experiment with different ranks
3. Try fine-tuning on your own data
4. Compare with full fine-tuning results