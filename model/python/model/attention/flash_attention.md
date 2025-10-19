# Flash Attention: A Beginner's Guide

## What is Attention?

Think of attention like reading a sentence and deciding which words are most important for understanding each word's meaning.

**Example:** In "The cat sat on the mat"
- When processing "sat", you pay attention to "cat" (who sat?) and "mat" (where?)
- Each word looks at all other words to understand context

## The Memory Problem with Standard Attention

Standard attention has a **quadratic memory problem**:

```
Sequence length: 1000 tokens
Attention matrix: 1000 × 1000 = 1,000,000 values
Sequence length: 10000 tokens  
Attention matrix: 10000 × 10000 = 100,000,000 values (100x more memory!)
```

**Real example:**
- GPT-3 context: 2048 tokens → 4M attention values per head
- Long document: 8192 tokens → 67M attention values per head
- With 96 attention heads → 6.4 billion values to store!

## What is Flash Attention?

Flash Attention solves this by **never storing the full attention matrix**. Instead, it:
1. Processes attention in small blocks
2. Computes the same mathematical result
3. Uses clever bookkeeping to maintain accuracy

## Visual Example: 8 Tokens with Block Size 4

### Standard Attention (Memory Intensive)
```
Tokens: ["The", "cat", "sat", "on", "the", "big", "red", "mat"]

Full Attention Matrix (8×8 = 64 values stored):
     The  cat  sat  on   the  big  red  mat
The  [0.1 0.2 0.1 0.05 0.3 0.1 0.05 0.1]
cat  [0.2 0.3 0.2 0.1  0.1 0.05 0.0 0.05]
sat  [0.1 0.4 0.2 0.15 0.1 0.0  0.0 0.05]
on   [0.05 0.1 0.3 0.2 0.25 0.05 0.0 0.05]
the  [0.2 0.1 0.1 0.2 0.2 0.1 0.05 0.05]
big  [0.1 0.05 0.0 0.05 0.1 0.3 0.3 0.1]
red  [0.05 0.0 0.0 0.0 0.05 0.4 0.4 0.1]
mat  [0.1 0.1 0.1 0.1 0.1 0.2 0.2 0.1]
```

### Flash Attention (Memory Efficient)
```
Process in 4×4 blocks, never store full matrix:

Block 1: Tokens 0-3 attending to tokens 0-3
Block 2: Tokens 0-3 attending to tokens 4-7  
Block 3: Tokens 4-7 attending to tokens 0-3
Block 4: Tokens 4-7 attending to tokens 4-7

Only store one 4×4 block at a time (16 values vs 64)!
```

## Code Walkthrough: Simple Example

Let's trace through a minimal example:

```python
# Input: 4 tokens, embedding dim 8, block size 2
tokens = 4
embed_dim = 8
block_size = 2

# Q, K, V matrices (normally from neural networks)
Q = [[1, 0, 1, 0, 1, 0, 1, 0],  # Token 0
     [0, 1, 0, 1, 0, 1, 0, 1],  # Token 1  
     [1, 1, 0, 0, 1, 1, 0, 0],  # Token 2
     [0, 0, 1, 1, 0, 0, 1, 1]]  # Token 3

K = Q  # Same as Q for simplicity
V = Q  # Same as Q for simplicity
```

### Standard Attention (What We Want to Avoid)
```python
# This creates a 4×4 matrix - memory intensive!
attention_matrix = Q @ K.T  # Shape: (4, 4)
weights = softmax(attention_matrix)
output = weights @ V
```

### Flash Attention (Memory Efficient)
```python
# Process in 2×2 blocks instead
for i in range(0, 4, 2):  # Query blocks: [0,1], [2,3]
    q_block = Q[i:i+2]    # Shape: (2, 8)
    
    for j in range(0, 4, 2):  # Key/Value blocks: [0,1], [2,3]
        k_block = K[j:j+2]    # Shape: (2, 8)  
        v_block = V[j:j+2]    # Shape: (2, 8)
        
        # Only compute 2×2 attention block
        scores = q_block @ k_block.T  # Shape: (2, 2) - Much smaller!
        # ... process this block ...
```

## The Online Softmax Algorithm

The tricky part is computing softmax across blocks. Here's why:

### Problem: Softmax Needs Global Information
```python
# Standard softmax needs to see ALL values
scores = [2, 1, 4, 3]
softmax_result = [0.12, 0.04, 0.88, 0.32]  # Sums to 1.0
```

### Solution: Online Softmax
```python
# Process scores in blocks, maintaining running statistics
block1 = [2, 1]  # First block
block2 = [4, 3]  # Second block

# Block 1: 
max1 = 2, sum1 = exp(2-2) + exp(1-2) = 1 + 0.37 = 1.37

# Block 2:
max2 = 4, combined_max = max(2, 4) = 4
# Rescale block 1: multiply by exp(2-4) = 0.135
# New sum = 1.37 * 0.135 + exp(4-4) + exp(3-4) = 0.185 + 1 + 0.37 = 1.555

# Final softmax: [0.12, 0.04, 0.88, 0.32] ✓ Same result!
```

## Your Code Explained

Let's break down the key parts of your FlashAttention class:

### 1. Block Processing Loop
```python
for i in range(0, num_tokens, self.block_size):  # Query blocks
    q_block = Q[:, i:i+self.block_size]
    
    for j in range(0, num_tokens, self.block_size):  # Key/Value blocks
        k_block = K[:, j:j+self.block_size]
        v_block = V[:, j:j+self.block_size]
```
**What it does:** Instead of processing all tokens at once, process in small chunks.

### 2. Online Softmax State
```python
row_max = np.full((b, q_block.shape[1]), -np.inf)  # Track maximum
row_sum = np.zeros((b, q_block.shape[1]))           # Track sum
```
**What it does:** Keep running statistics to compute correct softmax.

### 3. Statistics Update
```python
new_max = np.maximum(row_max[:, :, None], block_max)
exp_prev = np.exp(row_max[:, :, None] - new_max)
row_sum = row_sum[:, :, None] * exp_prev + np.sum(exp_scores, axis=2, keepdims=True)
```
**What it does:** Update softmax statistics as new blocks are processed.

## Memory Comparison

### Standard Attention
```
Sequence length: N
Memory usage: O(N²)
Example: 8192 tokens = 67M values stored
```

### Flash Attention  
```
Sequence length: N
Block size: B
Memory usage: O(N + B²)
Example: 8192 tokens, 64 block size = 8192 + 4096 = 12K values stored
Reduction: 67M → 12K (5000x less memory!)
```

## When to Use Flash Attention

**Use Flash Attention when:**
- Processing long sequences (>1000 tokens)
- Limited GPU memory
- Training large models
- Need exact same results as standard attention

**Standard attention is fine for:**
- Short sequences (<512 tokens)  
- Plenty of memory available
- Prototyping/debugging

## Key Takeaways

1. **Same Math, Less Memory:** Flash Attention computes identical results to standard attention
2. **Block Processing:** Never materializes the full attention matrix
3. **Online Softmax:** Clever algorithm maintains softmax accuracy across blocks
4. **Scalability:** Enables processing of much longer sequences
5. **Practical Impact:** Makes large language models feasible on consumer hardware

The genius of Flash Attention is that it's mathematically equivalent to standard attention but with dramatically better memory efficiency - allowing us to process much longer sequences that would otherwise be impossible.