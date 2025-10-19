# Multi-Head Latent Attention

## Overview
Multi-Head Latent Attention compresses the input sequence into a smaller latent space before computing attention, reducing computational complexity from O(n²) to O(n×m) where m << n.

## Key Concept
Instead of every token attending to every other token, tokens attend to a compressed set of "latent tokens" that summarize the entire sequence.

## Architecture Diagram
```
Input Sequence (n tokens)     Latent Space (m tokens)     Output (n tokens)
┌─────┬─────┬─────┬─────┐    ┌─────┬─────┐              ┌─────┬─────┬─────┬─────┐
│ T1  │ T2  │ T3  │ T4  │───▶│ L1  │ L2  │─────────────▶│ O1  │ O2  │ O3  │ O4  │
│512d │512d │512d │512d │    │512d │512d │              │512d │512d │512d │512d │
└─────┴─────┴─────┴─────┘    └─────┴─────┘              └─────┴─────┴─────┴─────┘
     4 tokens                   2 latents                    4 tokens
```

## How It Works

### 1. Latent Compression
```python
# Compress input sequence to latent space
self.latent_tokens = Linear(d_in, latent_dim * d_out, bias=False)
latent = self.latent_tokens.forward(x)  # (batch, tokens, embed) → (batch, latent_dim * embed)
latent = reshape(latent, (b, self.latent_dim, self.d_out))  # → (batch, latent_dim, embed)
```

**What happens:**
- Input: (1, 1024, 768) - 1024 tokens, 768 dimensions each
- Linear layer learns to compress to: (1, 64, 768) - 64 latent tokens
- Each latent token summarizes information from all input tokens

### 2. Cross-Attention
```python
# Queries from original input, Keys/Values from latents
Q = self.query.forward(x)        # From original tokens
K = self.key.forward(latent)     # From compressed latents  
V = self.value.forward(latent)   # From compressed latents
```

**Attention computation:**
- Q: (1024, 768) - queries from all original tokens
- K,V: (64, 768) - keys/values from compressed latents
- Attention matrix: (1024 × 64) instead of (1024 × 1024)

## Complexity Comparison

| Method | Attention Matrix Size | Complexity |
|--------|----------------------|------------|
| Standard Attention | n × n | O(n²) |
| Latent Attention | n × m | O(n×m) |

**Example with 1024 tokens, 64 latents:**
- Standard: 1024² = 1,048,576 operations
- Latent: 1024 × 64 = 65,536 operations
- **16x reduction** in computational cost

## Concrete Example

### Input
```
Sentence: "The quick brown fox jumps over the lazy dog today"
Tokens: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "today"]
Shape: (1, 10, 512)
```

### Latent Compression (10 → 3 latents)
```python
# Linear layer learns to create 3 summary tokens
latent_tokens = compress(input_tokens)  # (1, 10, 512) → (1, 3, 512)

# Latent tokens might represent:
# L1: "Subject information" (The quick brown fox)
# L2: "Action information" (jumps over) 
# L3: "Object/context information" (the lazy dog today)
```

### Cross-Attention
```python
# Each original token attends to the 3 latent summaries
Q = ["The", "quick", "brown", ...]  # 10 queries
K,V = [L1, L2, L3]                  # 3 keys/values

# Attention matrix: 10×3 instead of 10×10
# "fox" might attend strongly to L1 (subject) and L2 (action)
# "jumps" might attend strongly to L2 (action)
```

## Benefits

1. **Computational Efficiency**: O(n×m) vs O(n²)
2. **Memory Efficiency**: Smaller attention matrices
3. **Long Sequence Handling**: Enables processing of very long sequences
4. **Learned Compression**: Model learns optimal summarization

## Trade-offs

1. **Information Loss**: Compression may lose fine-grained details
2. **Training Complexity**: Requires learning good latent representations
3. **Hyperparameter Sensitivity**: Latent dimension choice is critical

## Use Cases

- **Long Document Processing**: Legal documents, research papers
- **Video Understanding**: Long video sequences
- **Genomic Sequences**: DNA/protein analysis
- **Time Series**: Long temporal sequences

## Implementation Notes

```python
class MultiHeadLatentAttention:
    def __init__(self, d_in, d_out, num_heads, latent_dim):
        # Compression layer
        self.latent_tokens = Linear(d_in, latent_dim * d_out)
        
        # Standard attention components
        self.query = Linear(d_in, d_out)
        self.key = Linear(d_out, d_out)
        self.value = Linear(d_out, d_out)
        
    def forward(self, x):
        # 1. Compress to latent space
        latent = self.latent_tokens(x)
        latent = reshape(latent, (batch, latent_dim, d_out))
        
        # 2. Cross-attention: Q from input, K,V from latents
        Q = self.query(x)
        K = self.key(latent)
        V = self.value(latent)
        
        # 3. Compute attention with reduced complexity
        attention = softmax(Q @ K.T / sqrt(d_k)) @ V
        return attention
```

## Key Insight
The "magic" is in the learned compression - the model discovers which combinations of input information are most important to preserve in the latent space, creating efficient summaries that maintain task performance while dramatically reducing computational cost.