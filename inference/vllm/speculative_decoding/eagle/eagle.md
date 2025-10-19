# Eagle Speculative Decoding: A Beginner's Implementation Guide

## What is Eagle?

Eagle is a speculative decoding method that creates a **lightweight draft model** by performing "model surgery" on a large language model:

- **Keep**: Token embeddings and language model head
- **Replace**: Entire transformer stack with a simple MLP
- **Result**: Fast draft model that shares vocabulary with the original

## How Eagle Works

### The Architecture
```
Original Model: Embeddings → Transformer (24+ layers) → LM Head
Eagle Model:   Embeddings → MLP (2-3 layers) → LM Head
```

### The Process
1. **Draft**: Eagle MLP quickly generates candidate tokens
2. **Verify**: Original model checks candidates in parallel
3. **Accept**: Use verified tokens, reject bad ones

## Step-by-Step Implementation

### Step 1: Create the Eagle Model

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class EagleModel(nn.Module):
    def __init__(self, original_model, hidden_size=4096, mlp_layers=3):
        super().__init__()
        
        # Keep original components
        self.embeddings = original_model.model.embed_tokens
        self.lm_head = original_model.lm_head
        
        # Create lightweight MLP replacement
        layers = []
        for i in range(mlp_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            ])
        self.mlp = nn.Sequential(*layers)
        
        # Freeze original components during training
        for param in self.embeddings.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        # Get embeddings (frozen)
        hidden_states = self.embeddings(input_ids)
        
        # Process through MLP (trainable)
        hidden_states = self.mlp(hidden_states)
        
        # Get logits (frozen)
        logits = self.lm_head(hidden_states)
        return logits

# Create Eagle model
def create_eagle_model(model_name):
    original_model = AutoModel.from_pretrained(model_name)
    eagle_model = EagleModel(original_model)
    return eagle_model, original_model
```

### Step 2: Training Data Generation

```python
def generate_training_data(original_model, tokenizer, texts, max_length=512):
    """Generate training data by running original model on text corpus"""
    training_data = []
    
    original_model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            
            # Get target logits from original model
            outputs = original_model(**inputs)
            target_logits = outputs.logits
            
            training_data.append({
                'input_ids': inputs['input_ids'],
                'target_logits': target_logits
            })
    
    return training_data

# Example usage
texts = [
    "The weather is nice today.",
    "Python is a programming language.",
    # Add your training corpus here
]

training_data = generate_training_data(original_model, tokenizer, texts)
```

### Step 3: Train the Eagle Model

```python
def train_eagle_model(eagle_model, training_data, epochs=10, lr=1e-4):
    optimizer = torch.optim.AdamW(eagle_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    eagle_model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in training_data:
            optimizer.zero_grad()
            
            # Forward pass through Eagle
            eagle_logits = eagle_model(batch['input_ids'])
            
            # Compare with original model's output
            loss = criterion(eagle_logits, batch['target_logits'])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_data):.4f}")
    
    return eagle_model

# Train the model
trained_eagle = train_eagle_model(eagle_model, training_data)

# Save the trained Eagle model
torch.save(trained_eagle.state_dict(), "eagle_model.pt")
```

### Step 4: Convert for vLLM

```python
def save_eagle_for_vllm(eagle_model, tokenizer, save_path):
    """Save Eagle model in format compatible with vLLM"""
    
    # Save model weights
    eagle_model.save_pretrained(save_path)
    
    # Save tokenizer (same as original)
    tokenizer.save_pretrained(save_path)
    
    # Create config file
    config = {
        "model_type": "eagle",
        "hidden_size": 4096,
        "vocab_size": len(tokenizer),
        "mlp_layers": 3
    }
    
    import json
    with open(f"{save_path}/config.json", "w") as f:
        json.dump(config, f)

# Save for vLLM
save_eagle_for_vllm(trained_eagle, tokenizer, "./eagle-llama-7b")
```

### Step 5: Use with vLLM

```python
from vllm import LLM, SamplingParams

# Initialize with Eagle speculative decoding
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",        # Original target model
    speculative_model="./eagle-llama-7b",          # Your trained Eagle model
    num_speculative_tokens=4,                      # Number of tokens to speculate
    use_v2_block_manager=True,
    speculative_draft_tensor_parallel_size=1
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    top_p=0.9
)

# Generate with Eagle acceleration
prompt = "The future of AI is"
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

## Complete Training Script

```python
#!/usr/bin/env python3
"""
Complete Eagle model training script
"""
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

def main():
    # Configuration
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    SAVE_PATH = "./eagle-llama-7b"
    
    # Load original model and tokenizer
    print("Loading original model...")
    original_model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create Eagle model
    print("Creating Eagle model...")
    eagle_model = EagleModel(original_model)
    
    # Load training data (use your own dataset)
    print("Loading training data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
    texts = [item['text'] for item in dataset if len(item['text']) > 50]
    
    # Generate training data
    print("Generating training data...")
    training_data = generate_training_data(original_model, tokenizer, texts)
    
    # Train Eagle model
    print("Training Eagle model...")
    trained_eagle = train_eagle_model(eagle_model, training_data)
    
    # Save for vLLM
    print("Saving Eagle model...")
    save_eagle_for_vllm(trained_eagle, tokenizer, SAVE_PATH)
    
    print(f"Eagle model saved to {SAVE_PATH}")
    print("Ready to use with vLLM!")

if __name__ == "__main__":
    main()
```

## Performance Expectations

### Speed Improvements
- **2-4x faster** than standard generation
- **10-50x faster** draft generation vs full model
- **High acceptance rate** (60-80%) due to shared components

### Memory Usage
- **Eagle model**: ~10-20% of original model size
- **Total memory**: Original + Eagle (still less than 2 full models)
- **Efficient**: Shared embeddings and LM head

## Troubleshooting

### Low Acceptance Rate
- **Increase training epochs**: More training = better approximation
- **Better training data**: Use domain-specific data
- **Tune MLP architecture**: Try different layer sizes

### Slow Training
- **Reduce training data**: Start with smaller corpus
- **Use gradient accumulation**: For larger effective batch sizes
- **Mixed precision**: Use torch.cuda.amp for faster training

### Memory Issues
- **Smaller MLP**: Reduce hidden dimensions
- **Gradient checkpointing**: Trade compute for memory
- **Batch size**: Reduce if running out of memory

## Limitations

### Model-Specific
- Each target model needs its own Eagle variant
- Can't reuse Eagle across different model families

### Training Required
- Significant compute needed for training
- Need large, diverse training corpus
- Time-intensive process

### Quality Trade-offs
- MLP can't capture all transformer complexity
- May struggle with very creative or complex tasks
- Best for structured, predictable text

## Getting Started Checklist

1. **Choose target model**: Start with smaller models (7B) for testing
2. **Prepare training data**: Collect diverse, high-quality text corpus
3. **Set up training environment**: GPU with sufficient memory
4. **Train Eagle model**: Allow several hours for training
5. **Test with vLLM**: Verify performance improvements
6. **Tune parameters**: Optimize for your specific use case

Eagle offers significant speedups but requires upfront investment in training. Start with a small model and limited data to understand the process before scaling up!