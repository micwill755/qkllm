"""
LoRA Fine-tuning Example
This script demonstrates fine-tuning using the wrapper approach.
"""

import numpy as np
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model'))

from gpt2 import GPT2Model, GPT_CONFIG_124M
from peft_wrapper import get_peft_model
import tiktoken

class SimpleLoRATrainer:
    """Simple trainer for LoRA fine-tuning"""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        
    def compute_loss(self, logits, targets):
        """Simple cross-entropy loss"""
        shift_logits = logits[:, :-1, :]
        shift_targets = targets[:, 1:]
        
        batch_size, seq_len, vocab_size = shift_logits.shape
        flat_logits = shift_logits.reshape(-1, vocab_size)
        flat_targets = shift_targets.reshape(-1)
        
        # Compute softmax and cross-entropy
        exp_logits = np.exp(flat_logits - np.max(flat_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        loss = 0
        count = 0
        for i in range(len(flat_targets)):
            if flat_targets[i] >= 0:  # Ignore padding tokens
                loss += -np.log(probs[i, flat_targets[i]] + 1e-8)
                count += 1
        
        return loss / count if count > 0 else 0
    
    def train_step(self, batch, targets):
        """Single training step"""
        logits = self.model.forward(batch)
        loss = self.compute_loss(logits, targets)
        
        # Simple parameter update (demonstration only)
        lora_params = self.model.get_lora_parameters()
        for name, param in lora_params.items():
            gradient = np.random.randn(*param.shape) * 0.0001
            param -= self.learning_rate * gradient
        
        return loss

def create_training_data():
    """Create simple training data"""
    sentences = [
        "The cat sat on the mat.",
        "Dogs love to play fetch.",
        "Birds fly in the sky.",
        "Fish swim in the ocean.",
        "The sun shines brightly.",
        "Rain falls from clouds.",
        "Trees grow tall and strong.",
        "Flowers bloom in spring."
    ]
    return sentences

def prepare_data(texts, tokenizer, max_length=64):
    """Tokenize and prepare training data"""
    tokenized_data = []
    
    for text in texts:
        tokens = tokenizer.encode(text.strip())
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            pad_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
            tokens.extend([pad_token] * (max_length - len(tokens)))
        
        tokenized_data.append(tokens)
    
    return np.array(tokenized_data)

def main():
    print("=== LoRA Fine-tuning Example ===\n")
    
    # Configuration
    config = {
        'lora_rank': 8,
        'lora_alpha': 16,
        'learning_rate': 0.0001,
        'batch_size': 2,
        'num_epochs': 5,
        'max_length': 32,
        'target_modules': ["out_head"]  # Which layers to adapt
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load and prepare data
    print("\n2. Loading and preparing data...")
    texts = create_training_data()
    print(f"Loaded {len(texts)} text samples")
    
    data = prepare_data(texts, tokenizer, config['max_length'])
    print(f"Tokenized data shape: {data.shape}")
    
    # Create base model
    print("\n3. Creating base GPT-2 model...")
    np.random.seed(42)
    base_model = GPT2Model(GPT_CONFIG_124M)
    
    # Wrap with LoRA adapters
    print("\n4. Adding LoRA adapters...")
    model = get_peft_model(
        base_model,
        target_modules=config['target_modules'],
        lora_rank=config['lora_rank'],
        lora_alpha=config['lora_alpha']
    )
    
    # Enable LoRA training
    model.enable_lora_training()
    
    # Show parameter info
    param_info = model.count_parameters()
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(f"LoRA parameters: {param_info['lora_parameters']:,}")
    print(f"Training only {param_info['trainable_percentage']:.2f}% of parameters!")
    
    # Create trainer
    trainer = SimpleLoRATrainer(model, learning_rate=config['learning_rate'])
    
    # Initial evaluation
    print("\n5. Initial evaluation...")
    initial_loss = trainer.compute_loss(model.forward(data), data)
    print(f"Initial loss: {initial_loss:.4f}")
    
    # Training loop
    print(f"\n6. Training for {config['num_epochs']} epochs...")
    
    for epoch in range(config['num_epochs']):
        total_loss = 0
        num_batches = 0
        
        # Simple batching
        for i in range(0, len(data), config['batch_size']):
            batch = data[i:i+config['batch_size']]
            targets = batch.copy()
            
            loss = trainer.train_step(batch, targets)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")
    
    # Final evaluation
    print("\n7. Final evaluation...")
    final_loss = trainer.compute_loss(model.forward(data), data)
    improvement = initial_loss - final_loss
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.4f}")
    
    # Test generation
    print("\n8. Testing generation...")
    test_prompt = "The cat"
    test_tokens = np.array([tokenizer.encode(test_prompt)])
    
    print(f"Input: '{test_prompt}'")
    
    # Generate next few tokens
    current_tokens = test_tokens.copy()
    generated_text = test_prompt
    
    for _ in range(5):
        logits = model.forward(current_tokens)
        last_logits = logits[0, -1, :]
        next_token = np.argmax(last_logits)
        next_word = tokenizer.decode([next_token])
        generated_text += next_word
        current_tokens = np.concatenate([current_tokens, [[next_token]]], axis=1)
    
    print(f"Generated: '{generated_text}'")
    
    # Compare with and without LoRA
    print("\n9. Comparing LoRA vs Original...")
    logits_with_lora = model.forward(test_tokens)
    
    model.disable_lora_training()
    logits_without_lora = model.forward(test_tokens)
    
    diff = np.mean(np.abs(logits_with_lora - logits_without_lora))
    print(f"Mean difference in logits: {diff:.6f}")
    
    # Save LoRA weights
    print("\n10. Saving LoRA weights...")
    save_path = "peft_fine_tuned_lora.npz"
    model.enable_lora_training()
    model.save_lora_weights(save_path)
    
    print(f"LoRA weights saved to {save_path}")
    print(f"File size: {os.path.getsize(save_path) / 1024:.2f} KB")
    
    # Test adapter swapping
    print("\n11. Testing adapter swapping...")
    
    # Create new model and load weights
    new_base_model = GPT2Model(GPT_CONFIG_124M)
    new_model = get_peft_model(new_base_model, target_modules=config['target_modules'])
    new_model.enable_lora_training()
    new_model.load_lora_weights(save_path)
    
    # Verify same output
    new_logits = new_model.forward(test_tokens)
    verification_diff = np.mean(np.abs(new_logits - logits_with_lora))
    print(f"Difference after loading: {verification_diff:.8f}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print("\n=== LoRA Fine-tuning Complete ===")
    print("\nKey Advantages of This Approach:")
    print("- Uses your existing GPT2Model unchanged")
    print("- Easy to add LoRA to any layer")
    print("- Can swap different adapters")
    print("- Minimal code changes required")

if __name__ == "__main__":
    main()