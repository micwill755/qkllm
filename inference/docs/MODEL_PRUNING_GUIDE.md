# Model Pruning Guide

## Overview

Model pruning removes unnecessary weights from neural networks to reduce model size and improve inference speed while maintaining performance.

## Weight Identification Methods

### 1. Magnitude-Based Pruning
Remove weights with smallest absolute values:

```python
import torch
import torch.nn.utils.prune as prune

# L1 unstructured pruning (removes individual weights)
prune.l1_unstructured(module, name='weight', amount=0.3)  # Remove 30% smallest weights

# L2 (magnitude) pruning
prune.l2_unstructured(module, name='weight', amount=0.3)
```

### 2. Gradient-Based Pruning
Remove weights with minimal impact on loss:

```python
def gradient_based_pruning(model, dataloader, prune_ratio=0.3):
    model.train()
    for batch in dataloader:
        loss = compute_loss(model, batch)
        loss.backward()
    
    # Calculate importance scores
    importance_scores = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            importance_scores[name] = param.grad.abs()
    
    # Prune weights with lowest importance
    threshold = torch.quantile(torch.cat([scores.flatten() for scores in importance_scores.values()]), prune_ratio)
    
    for name, param in model.named_parameters():
        mask = importance_scores[name] > threshold
        param.data *= mask
```

### 3. Structured vs Unstructured Pruning

**Unstructured**: Remove individual weights
```python
# Creates sparse tensors, needs specialized hardware for speedup
prune.random_unstructured(module, name='weight', amount=0.5)
```

**Structured**: Remove entire channels/heads/layers
```python
# Removes entire neurons/channels - works on standard hardware
prune.random_structured(module, name='weight', amount=0.3, dim=0)  # Remove 30% of output channels
```

## Advanced Pruning Methods

### SNIP (Single-shot Network Pruning)
```python
def snip_pruning(model, dataloader, sparsity=0.9):
    for batch in dataloader:
        loss = compute_loss(model, batch)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Sensitivity = |weight * gradient|
        sensitivities = []
        for param, grad in zip(model.parameters(), grads):
            sensitivities.append((param * grad).abs())
        
        # Keep top (1-sparsity) fraction of weights
        all_scores = torch.cat([s.flatten() for s in sensitivities])
        threshold = torch.quantile(all_scores, sparsity)
        
        # Apply masks
        for param, sensitivity in zip(model.parameters(), sensitivities):
            mask = sensitivity > threshold
            param.data *= mask
```

### Lottery Ticket Hypothesis
```python
def lottery_ticket_pruning(model, train_fn, prune_ratio=0.2, iterations=5):
    # Save initial weights
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}
    
    for i in range(iterations):
        # Train model
        train_fn(model)
        
        # Prune lowest magnitude weights
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_ratio)
        
        # Reset remaining weights to initial values
        for name, param in model.named_parameters():
            if name in initial_weights:
                mask = getattr(model.get_submodule(name.split('.')[0]), name.split('.')[-1] + '_mask')
                param.data = initial_weights[name] * mask
```

## Practical Implementation

### Transformer Model Pruning
```python
def prune_transformer_model(model, method='magnitude', sparsity=0.3):
    """Prune a transformer model using specified method"""
    
    if method == 'magnitude':
        # Focus on attention and MLP layers
        for name, module in model.named_modules():
            if 'attention' in name or 'mlp' in name:
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
    
    elif method == 'structured':
        # Remove entire attention heads or MLP neurons
        for name, module in model.named_modules():
            if 'attention.out' in name:  # Output projection
                prune.ln_structured(module, name='weight', amount=sparsity, n=2, dim=0)
            elif 'mlp.dense' in name:  # MLP layers
                prune.ln_structured(module, name='weight', amount=sparsity, n=2, dim=1)
    
    # Remove pruning reparameterization to make permanent
    for module in model.modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
    
    return model
```

### Gradual Pruning Schedule
```python
def gradual_pruning(model, initial_sparsity=0.0, final_sparsity=0.9, num_steps=100):
    """Gradually increase pruning over training steps"""
    
    for step in range(num_steps):
        # Calculate current sparsity
        current_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * (step / num_steps)
        
        # Apply pruning
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=current_sparsity)
        
        # Train for one step
        train_step(model)
        
        # Remove and reapply pruning for next iteration
        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
```

## Pruning Strategies by Layer Type

### Attention Layers
```python
def prune_attention_layers(model, head_pruning_ratio=0.25, projection_pruning_ratio=0.1):
    """Prune attention mechanisms"""
    
    for name, module in model.named_modules():
        if 'self_attn' in name:
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                # Light pruning for query/key/value projections
                prune.l1_unstructured(module, name='weight', amount=projection_pruning_ratio)
            elif 'out_proj' in name:
                # Structured pruning for output projection (remove heads)
                prune.ln_structured(module, name='weight', amount=head_pruning_ratio, n=2, dim=0)
```

### MLP Layers
```python
def prune_mlp_layers(model, sparsity=0.3):
    """Prune feed-forward layers"""
    
    for name, module in model.named_modules():
        if 'mlp' in name and isinstance(module, torch.nn.Linear):
            if 'gate_proj' in name or 'up_proj' in name:
                # Higher pruning for intermediate layers
                prune.l1_unstructured(module, name='weight', amount=sparsity * 1.5)
            elif 'down_proj' in name:
                # Lower pruning for output projection
                prune.l1_unstructured(module, name='weight', amount=sparsity * 0.5)
```

## Evaluation and Fine-tuning

### Performance Recovery
```python
def prune_and_finetune(model, train_dataloader, val_dataloader, sparsity=0.3):
    """Complete pruning pipeline with fine-tuning"""
    
    # 1. Baseline evaluation
    baseline_acc = evaluate_model(model, val_dataloader)
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # 2. Apply pruning
    prune_transformer_model(model, method='magnitude', sparsity=sparsity)
    
    # 3. Evaluate after pruning
    pruned_acc = evaluate_model(model, val_dataloader)
    print(f"Accuracy after pruning: {pruned_acc:.4f}")
    
    # 4. Fine-tune to recover performance
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(3):  # Light fine-tuning
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
    
    # 5. Final evaluation
    final_acc = evaluate_model(model, val_dataloader)
    print(f"Accuracy after fine-tuning: {final_acc:.4f}")
    
    return model
```

## Pruning Impact Analysis

| Pruning Method | Sparsity | Memory Reduction | Speed Improvement | Quality Impact |
|----------------|----------|------------------|-------------------|----------------|
| Magnitude (Unstructured) | 50% | 1.2x | 1.1x* | Minimal |
| Magnitude (Unstructured) | 90% | 1.5x | 1.2x* | Small |
| Structured (Channels) | 25% | 1.3x | 1.4x | Small |
| Structured (Heads) | 50% | 1.8x | 2.1x | Moderate |
| SNIP | 95% | 2.0x | 1.3x* | Small |
| Lottery Ticket | 80% | 1.6x | 1.2x* | Minimal |

*Requires sparse tensor support for actual speedup

## Best Practices

### 1. Layer Sensitivity Analysis
```python
def analyze_layer_sensitivity(model, dataloader):
    """Determine which layers are most sensitive to pruning"""
    
    sensitivities = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Temporarily prune 10% of weights
            prune.l1_unstructured(module, name='weight', amount=0.1)
            
            # Measure performance drop
            accuracy = evaluate_model(model, dataloader)
            sensitivities[name] = accuracy
            
            # Remove pruning
            prune.remove(module, 'weight')
    
    return sensitivities
```

### 2. Progressive Pruning
- Start with low sparsity (10-20%)
- Gradually increase over training
- Fine-tune after each pruning step
- Monitor performance degradation

### 3. Hardware Considerations
- **Unstructured pruning**: Requires sparse tensor libraries (cuSPARSE, TensorRT)
- **Structured pruning**: Works on standard hardware immediately
- **Block-sparse**: Compromise between flexibility and hardware efficiency

### 4. Quality Preservation
- Prune less aggressive in early layers
- Preserve critical paths (residual connections)
- Use knowledge distillation during fine-tuning
- Consider task-specific importance metrics

## Integration with Other Optimizations

Pruning combines well with:
- **Quantization**: Prune first, then quantize remaining weights
- **Knowledge Distillation**: Use teacher model during pruning fine-tuning
- **Dynamic Inference**: Combine with early exit strategies