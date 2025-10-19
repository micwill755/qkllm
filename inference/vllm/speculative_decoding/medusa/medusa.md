# Medusa Speculative Decoding: A Beginner's Implementation Guide

## What is Medusa?

Medusa adds **multiple prediction heads** to an existing large language model to make **speculative guesses** about future tokens, eliminating the need for a separate draft model.

### The Key Insight
Instead of using a separate small model for drafting, Medusa adds lightweight "heads" that make **independent guesses** about what tokens might appear at different future positions based on current context.

### Important Clarification
**Medusa doesn't actually generate multiple tokens at once** - that would be impossible since future tokens depend on previous ones. Instead, it makes **educated guesses** about probable future tokens based on statistical patterns learned during training.

## How Medusa Works

### Architecture Comparison
```
Original Model:
Input → Transformer → Hidden States → LM Head → Next Token

Medusa Model:
Input → Transformer → Hidden States → LM Head → Next Token
                           ↓
                    Medusa Head 1 → Token +1
                    Medusa Head 2 → Token +2  
                    Medusa Head 3 → Token +3
```

### The Process
1. **Single Forward Pass**: Run input through the main transformer once
2. **Multiple Guesses**: Each Medusa head makes a guess about a different future position
3. **Verification**: Test each guess sequentially with the base model
4. **Accept/Reject**: Use verified guesses, reject incorrect ones

### How It Really Works
```
Current context: "The weather is"

Medusa Head 1: "What if the next token is 'nice'?"
Medusa Head 2: "What if the token after that is 'today'?" 
Medusa Head 3: "What if the token after that is 'and'?"
Medusa Head 4: "What if the token after that is 'sunny'?"

Result: Candidates = ["nice", "today", "and", "sunny"]

Verification:
1. Try "The weather is nice" → Base model: ✓ correct
2. Try "The weather is nice today" → Base model: ✓ correct  
3. Try "The weather is nice today and" → Base model: ✗ wrong

Accept: ["nice", "today"] | Reject: ["and", "sunny"]
```

## Understanding Medusa Training

### How Each Head Learns
```
Training Text: "The weather is nice today and sunny"

Training Examples:
Context: "The weather is" → Targets: [nice, today, and, sunny]
Context: "The weather is nice" → Targets: [today, and, sunny, tomorrow]
Context: "The weather is nice today" → Targets: [and, sunny, tomorrow, ...]
```

### What Each Head Learns
- **Head 1**: "After 'The weather is', the next token is often 'nice'"
- **Head 2**: "After 'The weather is', the second token is often 'today'"
- **Head 3**: "After 'The weather is', the third token is often 'and'"
- **Head 4**: "After 'The weather is', the fourth token is often 'sunny'"

### Why This Works
**Language has statistical patterns!** Given "The weather is", certain continuations are much more likely than others. Medusa heads learn these patterns and make educated guesses about probable continuations.

## Step-by-Step Implementation

### Step 1: Create Medusa Heads

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers=2):
        super().__init__()
        
        # Lightweight MLP for each head
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(hidden_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))
        
        # Final projection to vocabulary
        layers.append(nn.Linear(hidden_size, vocab_size))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, hidden_states):
        return self.layers(hidden_states)

class MedusaModel(nn.Module):
    def __init__(self, base_model, num_heads=4, hidden_size=4096):
        super().__init__()
        
        # Keep original model (frozen during Medusa training)
        self.base_model = base_model
        
        # Add multiple Medusa heads
        self.medusa_heads = nn.ModuleList([
            MedusaHead(hidden_size, base_model.config.vocab_size)
            for _ in range(num_heads)
        ])
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        # Get hidden states from base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Original prediction
        base_logits = outputs.logits
        
        # Medusa predictions for next k tokens
        medusa_logits = []
        for head in self.medusa_heads:
            head_logits = head(hidden_states)
            medusa_logits.append(head_logits)
        
        return {
            'base_logits': base_logits,
            'medusa_logits': medusa_logits,
            'hidden_states': hidden_states
        }
```

### Step 2: Training Data Generation

```python
def generate_medusa_training_data(model, tokenizer, texts, max_length=512):
    """Generate training data with multi-token targets"""
    training_data = []
    
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = inputs['input_ids'][0]
            
            # Create training examples
            for i in range(len(input_ids) - 4):  # Need at least 4 future tokens
                context = input_ids[:i+1]
                
                # Targets for each Medusa head
                targets = []
                for j in range(4):  # 4 heads predicting next 1,2,3,4 tokens
                    if i + j + 2 < len(input_ids):
                        targets.append(input_ids[i + j + 2])  # Next j+1 token
                    else:
                        targets.append(tokenizer.pad_token_id)
                
                training_data.append({
                    'input_ids': context.unsqueeze(0),
                    'targets': torch.tensor(targets)
                })
    
    return training_data

# Generate training data
texts = [
    "The weather is nice today. Tomorrow will be sunny.",
    "Python is a programming language. It is easy to learn.",
    # Add your training corpus
]

training_data = generate_medusa_training_data(base_model, tokenizer, texts)
```

### Step 3: Train Medusa Heads

```python
def train_medusa_heads(medusa_model, training_data, epochs=10, lr=1e-4):
    # Only train Medusa heads, base model stays frozen
    optimizer = torch.optim.AdamW(medusa_model.medusa_heads.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    medusa_model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in training_data:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = medusa_model(batch['input_ids'])
            
            # Compute loss for each Medusa head
            loss = 0
            for i, head_logits in enumerate(outputs['medusa_logits']):
                # Each head predicts the next i+1 token
                target = batch['targets'][i]
                if target != tokenizer.pad_token_id:
                    head_loss = criterion(head_logits[:, -1, :], target.unsqueeze(0))
                    loss += head_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_data):.4f}")
    
    return medusa_model

# Train Medusa heads
trained_medusa = train_medusa_heads(medusa_model, training_data)

# Save the trained model
torch.save(trained_medusa.state_dict(), "medusa_model.pt")
```

### Step 4: Inference with Medusa

```python
def medusa_generate(medusa_model, tokenizer, prompt, max_tokens=100):
    """Generate text using Medusa speculative decoding"""
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_tokens = []
    
    for _ in range(max_tokens):
        # Get predictions from all heads
        with torch.no_grad():
            outputs = medusa_model(input_ids)
            
            # Base model prediction (next token)
            base_logits = outputs['base_logits'][:, -1, :]
            next_token = torch.argmax(base_logits, dim=-1)
            
            # Medusa predictions (future tokens)
            candidates = [next_token.item()]
            for head_logits in outputs['medusa_logits']:
                candidate = torch.argmax(head_logits[:, -1, :], dim=-1)
                candidates.append(candidate.item())
            
            # Verify candidates with base model
            verified_tokens = verify_candidates(medusa_model.base_model, input_ids, candidates)
            
            # Accept verified tokens
            for token in verified_tokens:
                generated_tokens.append(token)
                input_ids = torch.cat([input_ids, torch.tensor([[token]])], dim=1)
                
                if token == tokenizer.eos_token_id:
                    break
    
    return tokenizer.decode(generated_tokens)

def verify_candidates(base_model, context, candidates):
    """Verify candidate tokens using the base model"""
    verified = []
    
    for i, candidate in enumerate(candidates):
        # Extend context with candidate
        test_input = torch.cat([context, torch.tensor([[candidate]])], dim=1)
        
        # Check if base model agrees
        with torch.no_grad():
            outputs = base_model(test_input)
            predicted = torch.argmax(outputs.logits[:, -2, :], dim=-1)
            
            if predicted.item() == candidate:
                verified.append(candidate)
                context = test_input  # Accept and continue
            else:
                break  # Reject this and all following candidates
    
    return verified
```

### Step 5: Convert for vLLM

```python
def save_medusa_for_vllm(medusa_model, tokenizer, save_path):
    """Save Medusa model for vLLM usage"""
    
    # Save the complete model
    medusa_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Create Medusa-specific config
    config = {
        "model_type": "medusa",
        "base_model": medusa_model.base_model.config.name_or_path,
        "num_medusa_heads": len(medusa_model.medusa_heads),
        "medusa_hidden_size": medusa_model.base_model.config.hidden_size,
        "medusa_num_layers": 2
    }
    
    import json
    with open(f"{save_path}/medusa_config.json", "w") as f:
        json.dump(config, f)

# Save for vLLM
save_medusa_for_vllm(trained_medusa, tokenizer, "./medusa-llama-7b")
```

### Step 6: Use with vLLM

```python
from vllm import LLM, SamplingParams

# Initialize with Medusa speculative decoding
llm = LLM(
    model="./medusa-llama-7b",           # Your trained Medusa model
    speculative_model=None,              # No separate draft model needed
    num_speculative_tokens=4,            # Number of Medusa heads
    use_v2_block_manager=True,
    speculative_draft_tensor_parallel_size=1
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    top_p=0.9
)

# Generate with Medusa acceleration
prompt = "The future of artificial intelligence"
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

## Complete Training Script

```python
#!/usr/bin/env python3
"""
Complete Medusa training script
"""
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

def main():
    # Configuration
    BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
    NUM_HEADS = 4
    SAVE_PATH = "./medusa-llama-7b"
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModel.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Create Medusa model
    print("Creating Medusa model...")
    medusa_model = MedusaModel(base_model, num_heads=NUM_HEADS)
    
    # Load training data
    print("Loading training data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
    texts = [item['text'] for item in dataset if len(item['text']) > 100]
    
    # Generate training data
    print("Generating training data...")
    training_data = generate_medusa_training_data(base_model, tokenizer, texts)
    
    # Train Medusa heads
    print("Training Medusa heads...")
    trained_medusa = train_medusa_heads(medusa_model, training_data)
    
    # Save for vLLM
    print("Saving Medusa model...")
    save_medusa_for_vllm(trained_medusa, tokenizer, SAVE_PATH)
    
    print(f"Medusa model saved to {SAVE_PATH}")
    print("Ready to use with vLLM!")

if __name__ == "__main__":
    main()
```

## Performance Characteristics

### Speed Improvements
- **3-5x faster** than standard generation
- **Higher acceptance rate** than separate draft models (70-90%)
- **Single forward pass** generates multiple token candidates

### Memory Usage
- **Minimal overhead**: Only adds small MLP heads
- **No separate model**: Uses existing transformer representations
- **Efficient**: ~5-10% memory increase vs base model

## Advantages over Other Methods

### vs N-gram Lookup
- **Better quality**: Uses learned representations, not just pattern matching
- **Context aware**: Understands semantic relationships
- **Consistent performance**: Works across different text types

### vs Eagle
- **No model surgery**: Keeps original model intact
- **Easier training**: Only train small heads, not entire MLP
- **Better integration**: Uses model's internal knowledge

### vs Separate Draft Model
- **Single model**: No need to manage two models
- **Better alignment**: Shares exact same representations
- **Higher acceptance**: Predictions are more consistent

## Limitations

### Training Requirements
- **Still needs training**: Must train Medusa heads on data
- **Model-specific**: Each base model needs its own Medusa heads
- **Quality dependent**: Performance depends on training data quality

### Architecture Constraints
- **Transformer-specific**: Works best with transformer models
- **Hidden state access**: Needs access to internal representations
- **Memory overhead**: Additional heads increase memory usage

## Getting Started Checklist

1. **Choose base model**: Start with smaller models for testing
2. **Prepare training data**: Collect diverse text corpus
3. **Train Medusa heads**: Focus on quality training data
4. **Test performance**: Verify speedup and quality
5. **Deploy with vLLM**: Integrate into production pipeline

Medusa offers an excellent balance between performance gains and implementation complexity - easier than Eagle but more sophisticated than n-gram lookup!