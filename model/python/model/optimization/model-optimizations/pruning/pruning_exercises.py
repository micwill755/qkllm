"""
Real-World GPT2 Pruning Techniques
==================================

Implement production-ready pruning methods used by companies like Google, Meta, and OpenAI.
Focus on techniques that are actually deployed in real systems.
"""

import numpy as np
from gpt2 import GPT2Model, GPT_CONFIG_124M
from linear import Linear
import copy

# ============================================================================
# EXERCISE 1: Magnitude-Based Pruning (Industry Standard)
# ============================================================================

def exercise_1_magnitude_pruning():
    """
    Exercise 1: Magnitude-Based Pruning (Used by TensorRT, PyTorch, TensorFlow)
    Goal: Implement the most common production pruning technique
    """
    print("=== EXERCISE 1: Magnitude-Based Pruning ===")
    
    # Create model
    np.random.seed(42)
    model = GPT2Model(GPT_CONFIG_124M)
    
    # Test input
    test_input = np.array([[1, 2, 3, 4]])  # Simple token sequence
    
    # Get baseline output
    baseline_output = model.forward(test_input)
    print(f"Baseline output shape: {baseline_output.shape}")
    print(f"Baseline first few logits: {baseline_output[0, 0, :5]}")
    
    # Prune the query weights of first attention layer
    def prune_layer_magnitude(layer, prune_ratio=0.3):
        """Remove smallest magnitude weights"""
        original_weights = layer.weight.copy()
        magnitudes = np.abs(original_weights)
        
        # Find threshold for pruning
        threshold = np.percentile(magnitudes, prune_ratio * 100)
        
        # Create mask (1 = keep, 0 = remove)
        mask = magnitudes >= threshold
        
        # Apply pruning
        layer.weight = original_weights * mask
        
        pruned_count = np.sum(mask == 0)
        total_count = mask.size
        print(f"  Pruned {pruned_count}/{total_count} weights ({100*pruned_count/total_count:.1f}%)")
        
        return mask
    
    # Prune first attention layer
    print("Pruning first attention query layer (30% sparsity)...")
    mask = prune_layer_magnitude(model.blocks[0].att.query, prune_ratio=0.3)
    
    # Test after pruning
    pruned_output = model.forward(test_input)
    print(f"Pruned first few logits: {pruned_output[0, 0, :5]}")
    
    # Calculate difference
    diff = np.mean(np.abs(baseline_output - pruned_output))
    print(f"Mean absolute difference: {diff:.6f}")
    
    return model, mask

# ============================================================================
# EXERCISE 2: Structured Pruning (Hardware-Optimized)
# ============================================================================

def exercise_2_structured_pruning():
    """
    Exercise 2: Structured Pruning (Used by Mobile/Edge Deployment)
    Goal: Remove entire neurons for immediate hardware speedup
    """
    print("\n=== EXERCISE 2: Structured Pruning ===")
    
    np.random.seed(42)
    model = GPT2Model(GPT_CONFIG_124M)
    
    # Test input
    test_input = np.array([[1, 2, 3, 4]])
    baseline_output = model.forward(test_input)
    
    def prune_neurons_structured(layer, prune_ratio=0.25):
        """Remove entire output neurons based on L2 norm"""
        original_weights = layer.weight.copy()
        
        # Calculate L2 norm for each output neuron (row)
        neuron_norms = np.linalg.norm(original_weights, axis=1)
        
        # Find neurons to keep
        num_neurons = len(neuron_norms)
        num_to_keep = int(num_neurons * (1 - prune_ratio))
        keep_indices = np.argsort(neuron_norms)[-num_to_keep:]
        
        # Create new smaller weight matrix
        new_weights = original_weights[keep_indices, :]
        
        # Update layer (this is a simplified version - real implementation would need to handle dimensions)
        print(f"  Original shape: {original_weights.shape}")
        print(f"  New shape: {new_weights.shape}")
        print(f"  Removed {num_neurons - num_to_keep} neurons")
        
        return new_weights, keep_indices
    
    # Analyze feed-forward layer
    ff_layer = model.blocks[0].ff.linear1
    print("Analyzing feed-forward layer for structured pruning...")
    
    new_weights, kept_indices = prune_neurons_structured(ff_layer, prune_ratio=0.25)
    
    print(f"Kept neuron indices: {kept_indices[:10]}...")  # Show first 10
    
    return model

# ============================================================================
# EXERCISE 3: Gradual Pruning (Production Training)
# ============================================================================

def exercise_3_gradual_pruning():
    """
    Exercise 3: Gradual Pruning (Used by Google, Meta for LLM training)
    Goal: Implement sparsity scheduling during training
    """
    print("\n=== EXERCISE 3: Gradual Pruning ===")
    
    np.random.seed(42)
    model = GPT2Model(GPT_CONFIG_124M)
    
    test_input = np.array([[1, 2, 3, 4]])
    baseline_output = model.forward(test_input)
    
    def gradual_magnitude_pruning(model, final_sparsity=0.8, steps=5):
        """Gradually increase sparsity over training steps (production approach)"""
        
        current_sparsity = 0.0
        step_size = final_sparsity / steps
        
        print(f"Gradual pruning to {final_sparsity:.1%} over {steps} steps")
        
        for step in range(steps):
            current_sparsity += step_size
            
            # Apply magnitude pruning at current sparsity level
            for i, block in enumerate(model.blocks):
                layers = [block.att.query, block.att.key, block.att.value, 
                         block.att.out_proj, block.ff.linear1, block.ff.linear2]
                
                for layer in layers:
                    # Calculate threshold for this layer
                    magnitudes = np.abs(layer.weight)
                    threshold = np.percentile(magnitudes, current_sparsity * 100)
                    
                    # Apply mask
                    mask = magnitudes >= threshold
                    layer.weight *= mask
            
            # Test performance at this step
            step_output = model.forward(test_input)
            diff = np.mean(np.abs(baseline_output - step_output))
            
            print(f"  Step {step+1}: {current_sparsity:.1%} sparsity, performance diff: {diff:.6f}")
        
        return current_sparsity
    
    # Apply gradual pruning
    final_sparsity = gradual_magnitude_pruning(model, final_sparsity=0.7, steps=4)
    
    return model

# ============================================================================
# EXERCISE 4: Global Magnitude Pruning (Advanced Production)
# ============================================================================

def exercise_4_global_pruning():
    """
    Exercise 4: Global Magnitude Pruning (Used by NVIDIA, Hugging Face)
    Goal: Prune across entire model based on global weight importance
    """
    print("\n=== EXERCISE 4: Global Magnitude Pruning ===")
    
    np.random.seed(42)
    model = GPT2Model(GPT_CONFIG_124M)
    
    test_input = np.array([[1, 2, 3, 4]])
    baseline_output = model.forward(test_input)
    
    def global_magnitude_pruning(model, target_sparsity=0.5):
        """Prune weights globally across all layers based on magnitude"""
        
        # Collect all weights and their magnitudes
        all_weights = []
        layer_info = []
        
        for i, block in enumerate(model.blocks):
            layers = {
                'query': block.att.query,
                'key': block.att.key, 
                'value': block.att.value,
                'out_proj': block.att.out_proj,
                'ff1': block.ff.linear1,
                'ff2': block.ff.linear2
            }
            
            for layer_name, layer in layers.items():
                weights_flat = layer.weight.flatten()
                magnitudes = np.abs(weights_flat)
                
                all_weights.extend(magnitudes)
                layer_info.extend([(i, layer_name, layer, idx) for idx in range(len(weights_flat))])
        
        # Find global threshold
        all_weights = np.array(all_weights)
        threshold = np.percentile(all_weights, target_sparsity * 100)
        
        print(f"Global threshold: {threshold:.6f}")
        print(f"Total weights: {len(all_weights):,}")
        
        # Apply pruning globally
        pruned_count = 0
        for magnitude, (block_idx, layer_name, layer, weight_idx) in zip(all_weights, layer_info):
            if magnitude < threshold:
                # Convert flat index back to 2D coordinates
                row = weight_idx // layer.weight.shape[1]
                col = weight_idx % layer.weight.shape[1]
                layer.weight[row, col] = 0.0
                pruned_count += 1
        
        actual_sparsity = pruned_count / len(all_weights)
        print(f"Pruned {pruned_count:,}/{len(all_weights):,} weights ({actual_sparsity:.1%} sparsity)")
        
        return actual_sparsity
    
    # Apply global pruning
    sparsity = global_magnitude_pruning(model, target_sparsity=0.6)
    
    # Test performance
    pruned_output = model.forward(test_input)
    diff = np.mean(np.abs(baseline_output - pruned_output))
    print(f"Performance impact: {diff:.6f}")
    
    return model

# ============================================================================
# MAIN EXECUTION - Real-World Pruning Pipeline
# ============================================================================

if __name__ == "__main__":
    print("Real-World GPT2 Pruning Techniques")
    print("====================================")
    
    # Exercise 1: Magnitude-based pruning (most common)
    model1, mask = exercise_1_magnitude_pruning()
    
    # Exercise 2: Structured pruning (hardware-optimized)
    model2 = exercise_2_structured_pruning()
    
    # Exercise 3: Gradual pruning (training-time)
    model3 = exercise_3_gradual_pruning()
    
    # Exercise 4: Global pruning (advanced)
    model4 = exercise_4_global_pruning()
    
    print("\n" + "="*50)
    print("Production-Ready Pruning Techniques Completed!")
    print("\nThese methods are used by:")
    print("- Google (BERT, T5 optimization)")
    print("- Meta (LLaMA compression)")
    print("- NVIDIA (TensorRT optimization)")
    print("- Hugging Face (Model deployment)")
    print("- OpenAI (GPT model compression)")