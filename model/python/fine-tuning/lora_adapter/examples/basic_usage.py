"""
LoRA Usage Example
This script demonstrates how to use LoRA adapters with your existing GPT-2 model
using a wrapper approach (similar to Hugging Face PEFT).
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'model'))

from gpt2 import GPT2Model, GPT_CONFIG_124M
from peft_wrapper import get_peft_model
import tiktoken

def main():
    print("=== LoRA Usage Example ===\n")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create sample text
    texts = [
        "The future of AI is",
        "Machine learning will"
    ]
    
    # Tokenize
    batch = []
    eos_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    for text in texts:
        tokens = tokenizer.encode(text)
        batch.append(np.array(tokens))
    
    # Pad sequences to same length
    max_len = max(len(seq) for seq in batch)
    padded_batch = []
    for seq in batch:
        padded = np.pad(seq, (0, max_len - len(seq)), constant_values=eos_token)
        padded_batch.append(padded)
    
    batch = np.stack(padded_batch, axis=0)
    print(f"Input batch shape: {batch.shape}")
    print(f"Input texts: {texts}")
    
    # 1. Create your original GPT-2 model
    print("\n1. Creating original GPT-2 Model...")
    np.random.seed(42)
    base_model = GPT2Model(GPT_CONFIG_124M)
    
    # 2. Wrap with LoRA adapters (PEFT-style)
    print("\n2. Wrapping with LoRA adapters...")
    model = get_peft_model(
        base_model, 
        target_modules=["out_head"],  # Which layers to adapt
        lora_rank=16, 
        lora_alpha=16
    )
    
    # 3. Count parameters
    param_info = model.count_parameters()
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(f"LoRA parameters: {param_info['lora_parameters']:,}")
    print(f"Trainable percentage: {param_info['trainable_percentage']:.2f}%")
    
    # 4. Forward pass without LoRA
    print("\n3. Forward pass without LoRA adaptation...")
    model.disable_lora_training()
    logits_original = model.forward(batch)
    print(f"Output shape: {logits_original.shape}")
    print(f"Sample logits (first 5): {logits_original[0, 0, :5]}")
    
    # 5. Forward pass with LoRA
    print("\n4. Forward pass with LoRA adaptation...")
    model.enable_lora_training()
    logits_lora = model.forward(batch)
    print(f"Output shape: {logits_lora.shape}")
    print(f"Sample logits (first 5): {logits_lora[0, 0, :5]}")
    
    # 6. Compare outputs
    diff = np.mean(np.abs(logits_lora - logits_original))
    print(f"\nMean absolute difference: {diff:.6f}")
    print("(Small difference expected since LoRA B matrices start at zero)")
    
    # 7. Demonstrate LoRA parameter access
    print("\n5. LoRA Parameters Overview...")
    lora_params = model.get_lora_parameters()
    print(f"Number of LoRA parameter groups: {len(lora_params)}")
    
    # Show LoRA matrix shapes
    for name, param in lora_params.items():
        print(f"{name}: {param.shape}")
    
    # 8. Save and load LoRA weights
    print("\n6. Saving LoRA weights...")
    save_path = "peft_lora_weights.npz"
    model.save_lora_weights(save_path)
    
    # Modify LoRA weights slightly to test loading
    original_lora_A = model._get_module_by_name("out_head").lora_A.copy()
    model._get_module_by_name("out_head").lora_A += np.random.randn(*original_lora_A.shape) * 0.01
    
    logits_modified = model.forward(batch)
    diff_modified = np.mean(np.abs(logits_modified - logits_lora))
    print(f"Difference after LoRA modification: {diff_modified:.6f}")
    
    # Load original weights back
    print("\n7. Loading LoRA weights...")
    model.load_lora_weights(save_path)
    logits_restored = model.forward(batch)
    diff_restored = np.mean(np.abs(logits_restored - logits_lora))
    print(f"Difference after restoration: {diff_restored:.8f}")
    
    # 9. Demonstrate module restoration
    print("\n8. Demonstrating module restoration...")
    print("Current model has LoRA adapters")
    
    # Restore original modules
    model.restore_original_modules()
    logits_restored_original = base_model.forward(batch)
    diff_original = np.mean(np.abs(logits_restored_original - logits_original))
    print(f"Difference after module restoration: {diff_original:.8f}")
    print("(Should be very close to zero)")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print("\n=== LoRA Usage Example Complete ===")
    print("\nKey Benefits of This Approach:")
    print("- Works with your existing GPT2Model")
    print("- No need for separate model classes")
    print("- Easy to add/remove LoRA adapters")
    print("- Modular and flexible")

if __name__ == "__main__":
    main()