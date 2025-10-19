# Cross-Attention Explained

## Definition
Cross-attention is when **queries come from one sequence** and **keys/values come from a different sequence**. It enables information flow between different representations or modalities.

## Self-Attention vs Cross-Attention

| Type | Queries (Q) | Keys (K) | Values (V) | Purpose |
|------|-------------|----------|------------|---------|
| **Self-Attention** | Sequence A | Sequence A | Sequence A | Tokens attend to other tokens in same sequence |
| **Cross-Attention** | Sequence A | Sequence B | Sequence B | Sequence A attends to information in Sequence B |

## Visual Comparison

### Self-Attention
```
Sequence: ["The", "cat", "sits"]
    ↓       ↓       ↓
   Q,K,V   Q,K,V   Q,K,V
    ↓       ↓       ↓
Each token attends to all tokens in same sequence
```

### Cross-Attention
```
Sequence A: ["Le", "chat"]     →  Q (queries)
                ↓
            Cross-Attention
                ↓
Sequence B: ["The", "cat", "sits"]  →  K,V (keys/values)

French tokens ask: "Which English words should I focus on?"
```

## Common Use Cases

### 1. Machine Translation (Transformer Decoder)
```python
# Encoder-Decoder Cross-Attention
encoder_output = ["The", "cat", "sits"]  # English (K,V)
decoder_state = ["Le"]                   # French so far (Q)

# French token "Le" attends to English tokens to decide next word
attention_weights = softmax(Q @ K.T)  # [0.8, 0.1, 0.1] - focuses on "The"
next_word = "chat"  # Based on attending to English
```

### 2. Image Captioning
```python
image_features = [region1, region2, region3]  # CNN features (K,V)
text_tokens = ["A", "cat"]                    # Generated text (Q)

# Text tokens attend to image regions
# "cat" token might attend strongly to region containing the cat
```

### 3. Multi-Modal Models (CLIP, DALL-E)
```python
text_embedding = "A red car"     # Text representation (Q)
image_patches = [patch1, ...]    # Image patches (K,V)

# Text attends to relevant image patches
# "red" attends to red regions, "car" attends to car-shaped regions
```

## Cross-Attention in Multi-Head Latent Attention

```python
# Original tokens ask: "Which latent summaries are relevant to me?"
original_tokens = ["The", "quick", "brown", "fox"]  # Q (queries)
latent_summaries = [L1, L2]                         # K,V (keys/values)

# Cross-attention matrix: 4×2 instead of 4×4
Q = query_projection(original_tokens)    # (4, dim)
K = key_projection(latent_summaries)     # (2, dim)  
V = value_projection(latent_summaries)   # (2, dim)

attention_scores = Q @ K.T  # (4, 2) - each original token attends to 2 latents
```

## Concrete Example: Translation

### Input
- **English (Source)**: "The quick brown fox"
- **French (Target)**: "Le renard brun rapide"

### Cross-Attention Process
```
Step 1: Generate "Le"
French Q: [Le_query]
English K,V: ["The", "quick", "brown", "fox"]
Attention: Le focuses on "The" → generates "Le"

Step 2: Generate "renard" 
French Q: [renard_query]
English K,V: ["The", "quick", "brown", "fox"]  
Attention: renard focuses on "fox" → generates "renard"

Step 3: Generate "brun"
French Q: [brun_query]
English K,V: ["The", "quick", "brown", "fox"]
Attention: brun focuses on "brown" → generates "brun"
```

## Implementation Pattern

```python
def cross_attention(query_seq, key_value_seq):
    # Queries from sequence A
    Q = query_projection(query_seq)
    
    # Keys and Values from sequence B  
    K = key_projection(key_value_seq)
    V = value_projection(key_value_seq)
    
    # Cross-attention computation
    attention_scores = Q @ K.T / sqrt(d_k)
    attention_weights = softmax(attention_scores)
    output = attention_weights @ V
    
    return output
```

## Key Benefits

1. **Information Bridge**: Connects different modalities or representations
2. **Selective Focus**: Allows precise attention to relevant parts of other sequence
3. **Flexible Architecture**: Enables complex multi-modal and sequence-to-sequence models
4. **Interpretability**: Attention weights show what the model is "looking at"

## Real-World Applications

- **Google Translate**: Cross-attention between source and target languages
- **GPT-4 Vision**: Text tokens attend to image features
- **DALL-E**: Text descriptions attend to image generation process
- **Speech Recognition**: Audio features cross-attend with text predictions
- **Document QA**: Question tokens attend to document passages

## Key Insight
Cross-attention is the mechanism that allows AI models to **connect and relate information across different types of data**, enabling sophisticated multi-modal understanding and generation capabilities.