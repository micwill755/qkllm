import numpy as np
from model.python.model.lib.matrix_helper import transpose, mat_mul, create_mask, apply_mask, reshape, apply_mask_nd, transpose_nd, mat_mul_nd, combine_heads
from linear import Linear

def softmax(m):
    n_tokens_d1, n_tokens_d2 = m.shape
    out = np.zeros((n_tokens_d1, n_tokens_d2))

    for i in range(n_tokens_d1):
        max_val = np.max(m[i])
        sum = 0

        # first calculate the sum
        for j in range(n_tokens_d2):
            sum += np.exp(m[i][j] - max_val)
        
        # then divide each to get weight of 1
        for j in range(n_tokens_d2):
            out[i][j] = np.exp(m[i][j] - max_val) / sum

    return out

def softmax_nd(m):
    b, heads, n_tokens_d1, n_tokens_d2 = m.shape
    out = np.zeros((b, heads, n_tokens_d1, n_tokens_d2))

    for batch in range(b):
        for h in range(heads):
            out[batch][h] = softmax(m[batch][h])

    return out

'''
Self Attention:

- Each token can attend to ALL other tokens (including future ones)
- No masking applied
- Used in encoders (like BERT)
'''
class SelfAttention:
    def __init__ (self, d_in, d_out, context_length, dropout, qkv_bias=False):
        self.d_out = d_out
        self.d_in = d_in
        self.droput = dropout

        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_in, d_out, bias=qkv_bias)
        self.value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)

    def forward(self, x):
        b, num_tokens, embed_dim = x.shape

        query_w = self.query.forward(x)
        key_w = self.key.forward(x)
        value_w = self.value.forward(x)

        att_scores = mat_mul(query_w, transpose(key_w))
        attn_weights = softmax(att_scores)
        context = mat_mul(attn_weights, value_w)
        context = self.out_proj.forward(context)
        
        return context
            
'''
Causal Attention:

- Each token can only attend to previous tokens and itself
- Uses causal mask to block future tokens
- Used in decoders (like GPT)

'''

class CausalAttention:
    def __init__ (self, d_in, d_out, context_length, dropout, qkv_bias=False):
        self.d_out = d_out
        self.d_in = d_in
        self.droput = dropout

        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_in, d_out, bias=qkv_bias)
        self.value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)
        self.mask = create_mask(context_length, context_length)

    def forward(self, x):
        b, num_tokens, emd_dim = x.shape

        query_w = self.query.forward(x)
        key_w = self.key.forward(x)
        value_w = self.value.forward(x)

        att_scores = mat_mul(query_w, transpose(key_w))
        apply_mask(att_scores, self.mask)
        attn_weights = softmax(att_scores)
        context = mat_mul(attn_weights, value_w)
        context = self.out_proj.forward(context)
    
        return context

'''
Scaled dot product Attention:

Scaled Dot-Product Attention is the mathematical formula:
Attention(Q,K,V) = softmax(QK^T / √d_k)V

The key feature is scaling by √d_k (square root of key dimension) to prevent extremely large dot products.

Difference from Causal Attention:

Scaled Dot-Product: Refers to the mathematical operation (with scaling)
Causal Attention: Refers to the masking pattern (blocking future tokens)

'''
class ScaledDotProductAttention:
    def __init__ (self, d_in, d_out, context_length, dropout, qkv_bias=False):
        self.d_out = d_out
        self.d_in = d_in
        self.droput = dropout

        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_in, d_out, bias=qkv_bias)
        self.value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)
        self.mask = create_mask(context_length, context_length)

    def forward(self, x):
        b, num_tokens, emd_dim = x.shape

        query_w = self.query.forward(x)
        key_w = self.key.forward(x)
        value_w = self.value.forward(x)

        # we only add
        att_scores = mat_mul(query_w, transpose(key_w)) / np.sqrt(self.d_out)
        apply_mask(att_scores, self.mask)
        attn_weights = softmax(att_scores)
        context = mat_mul(attn_weights, value_w)
        context = self.out_proj.forward(context)
        
        return context
    

# temp for demonstration
class MultiHeadCausualAttention:
    def __init__ (self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        self.num_heads = num_heads
        self.heads = [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for i in range(num_heads)]

    def forward(self, x):
        results = []
        for i in range(self.num_heads):
            results.append(self.heads[i].forward(x))

        return results
    
'''

Multi-head attention splits the embedding dimensions across multiple parallel attention heads, each 
learning different relationships. For example, with input shape (batch, 4_tokens, 512_dims) and 8_heads: 
each head gets 512/8 = 64 dimensions per token, computes its own (4×4) attention matrix on those 64 dimensions, 
outputs (4_tokens, 64_dims), then all 8 head outputs are concatenated back to (4_tokens, 512_dims). 

This allows the model to simultaneously capture different types of relationships (like syntax in head 1, semantics in head 2) 
while maintaining the same overall dimensionality.

    Input: (batch, 4 tokens, 512 dims)
                    |
            Split into 8 heads
                    |
    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │Head1│Head2│Head3│Head4│Head5│Head6│Head7│Head8│
    │ 64  │ 64  │ 64  │ 64  │ 64  │ 64  │ 64  │ 64  │
    │dims │dims │dims │dims │dims │dims │dims │dims │
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
         |     |     |     |     |     |     |     |
    Each head computes (4×4) attention matrix
         |     |     |     |     |     |     |     |
    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │ 4×64│ 4×64│ 4×64│ 4×64│ 4×64│ 4×64│ 4×64│ 4×64│
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                    |
                Concatenate
                    |
        Output: (batch, 4 tokens, 512 dims)

'''

class MultiHeadAttention:
    def __init__ (self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        self.d_out = d_out
        self.d_in = d_in
        self.droput = dropout
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_in, d_out, bias=qkv_bias)
        self.value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)

        self.mask = create_mask(context_length, context_length)

    def forward(self, x):
        b, num_tokens, emd_dim = x.shape

        # temp until we handle parallel
        query_w = self.query.forward(x)
        key_w = self.key.forward(x)
        value_w = self.value.forward(x)

        # reshape full size attention matrices for q, k, v into [heads, tokens, embeddings]
        queries = reshape(query_w, (b, num_tokens, self.num_heads, self.head_dim))
        keys = reshape(key_w, (b, num_tokens, self.num_heads, self.head_dim))
        values = reshape(value_w, (b, num_tokens, self.num_heads, self.head_dim))

        # other attention implementations require this tranposition but we are going to cut it out 
        # we have to transpose the 3d tensor from [tokens, heads, embeddings] -> [heads, tokens, embeddings]
        #queries = transpose_nd(queries, 0, 1)
        #keys = transpose_nd(keys, 0, 1)
        #values = transpose_nd(values, 0, 1)

        att_scores = mat_mul_nd(queries, transpose_nd(keys, 2, 3)) / np.sqrt(self.head_dim)

        apply_mask_nd(att_scores, self.mask)
        attn_weights = softmax_nd(att_scores)
        context = mat_mul_nd(attn_weights, values)

        # other attention implementations require this tranposition but we are going to cut it out 
        # Shape: (b, num_tokens, num_heads, head_dim)
        #context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # we will then combine the heads to the original attention matrix [heads, tokens, head_dim] -> [tokens, heads * head_dim]
        combined = combine_heads(context)
        context = self.out_proj.forward(combined)
        return context


'''
Grouped Query Attention (GQA) reduces memory usage by sharing key and value projections 
across multiple query heads. Instead of having separate K,V for each head, query heads 
are grouped to share K,V projections.

For example, with 8 query heads and 2 KV heads:
- Query heads 0,1,2,3 share KV head 0
- Query heads 4,5,6,7 share KV head 1

    Query Heads (8):  ┌─Q1─┬─Q2─┬─Q3─┬─Q4─┬─Q5─┬─Q6─┬─Q7─┬─Q8─┐
                      └────┴────┴────┴────┴────┴────┴────┴────┘
                           \         /         \         /
                            \       /           \       /
    KV Heads (2):            ┌─KV1─┐             ┌─KV2─┐
                             └─────┘             └─────┘

This reduces KV cache size significantly while maintaining most performance benefits.
'''
class GroupedQueryAttention:
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, num_kv_heads, qkv_bias=False):
        self.d_out = d_out
        self.d_in = d_in
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_out // num_heads
        self.kv_head_dim = d_out // num_kv_heads
        self.group_size = num_heads // num_kv_heads
        
        # Q has full heads, K,V have fewer heads
        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_in, num_kv_heads * self.kv_head_dim, bias=qkv_bias)
        self.value = Linear(d_in, num_kv_heads * self.kv_head_dim, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)
        self.mask = create_mask(context_length, context_length)

    def forward(self, x):
        b, num_tokens, embed_dim = x.shape
        
        # Generate Q, K, V with different head counts
        Q = self.query.forward(x)
        K = self.key.forward(x) 
        V = self.value.forward(x)
        
        # Reshape Q to [batch, tokens, num_heads, head_dim]
        Q = reshape(Q, (b, num_tokens, self.num_heads, self.head_dim))
        
        # Reshape K,V to [batch, tokens, num_kv_heads, kv_head_dim]
        K = reshape(K, (b, num_tokens, self.num_kv_heads, self.kv_head_dim))
        V = reshape(V, (b, num_tokens, self.num_kv_heads, self.kv_head_dim))
        
        # Repeat K,V to match Q heads by group_size
        K_expanded = np.repeat(K, self.group_size, axis=2)
        V_expanded = np.repeat(V, self.group_size, axis=2)
        
        # Standard attention computation
        att_scores = mat_mul_nd(Q, transpose_nd(K_expanded, 2, 3)) / np.sqrt(self.head_dim)
        apply_mask_nd(att_scores, self.mask)
        attn_weights = softmax_nd(att_scores)
        context = mat_mul_nd(attn_weights, V_expanded)
        
        # Combine heads back to original dimension
        combined = combine_heads(context)
        return self.out_proj.forward(combined)

'''
Multi-Head Latent Attention compresses the key-value sequence into a smaller latent space
before computing attention, reducing computational complexity from O(n²) to O(n*m) where
m << n is the latent dimension.

    Input Sequence (n tokens)     Latent Space (m tokens)     Output (n tokens)
    ┌─────┬─────┬─────┬─────┐    ┌─────┬─────┐              ┌─────┬─────┬─────┬─────┐
    │ T1  │ T2  │ T3  │ T4  │───▶│ L1  │ L2  │─────────────▶│ O1  │ O2  │ O3  │ O4  │
    │512d │512d │512d │512d │    │512d │512d │              │512d │512d │512d │512d │
    └─────┴─────┴─────┴─────┘    └─────┴─────┘              └─────┴─────┴─────┴─────┘
         4 tokens                   2 latents                    4 tokens
         
    Cross-attention: Q from input, K,V from compressed latents
    Complexity: O(4×2) = O(8) instead of O(4×4) = O(16)
'''
class MultiHeadLatentAttention:
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, latent_dim, qkv_bias=False):
        self.d_out = d_out
        self.d_in = d_in
        self.dropout = dropout
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = d_out // num_heads
        
        # Latent compression
        self.latent_tokens = Linear(d_in, latent_dim * d_out, bias=False)
        
        # Standard attention projections
        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_out, d_out, bias=qkv_bias)
        self.value = Linear(d_out, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)

    def forward(self, x):
        b, num_tokens, embed_dim = x.shape
        
        # Compress input to latent space
        latent = self.latent_tokens.forward(x)
        latent = reshape(latent, (b, self.latent_dim, self.d_out))
        
        # Generate Q from original input, K,V from latents
        Q = self.query.forward(x)
        K = self.key.forward(latent)
        V = self.value.forward(latent)
        
        # Reshape for multi-head attention
        Q = reshape(Q, (b, num_tokens, self.num_heads, self.head_dim))
        K = reshape(K, (b, self.latent_dim, self.num_heads, self.head_dim))
        V = reshape(V, (b, self.latent_dim, self.num_heads, self.head_dim))
        
        # Cross-attention: input queries attend to latent keys/values
        att_scores = mat_mul_nd(Q, transpose_nd(K, 2, 3)) / np.sqrt(self.head_dim)
        attn_weights = softmax_nd(att_scores)
        context = mat_mul_nd(attn_weights, V)
        
        # Combine heads and project
        combined = combine_heads(context)
        return self.out_proj.forward(combined)

'''
Cross-Attention enables information flow between two different sequences. Queries come from 
one sequence while Keys and Values come from another sequence, allowing the first sequence 
to "attend to" or "look at" information from the second sequence.

    Sequence A (queries)          Sequence B (keys/values)        Output
    ┌─────┬─────┬─────┐          ┌─────┬─────┬─────┬─────┐      ┌─────┬─────┬─────┐
    │ Q1  │ Q2  │ Q3  │    ×     │ K1  │ K2  │ K3  │ K4  │  →  │ O1  │ O2  │ O3  │
    │512d │512d │512d │          │512d │512d │512d │512d │      │512d │512d │512d │
    └─────┴─────┴─────┘          └─────┴─────┴─────┴─────┘      └─────┴─────┴─────┘
         3 tokens                      4 tokens                    3 tokens
         
    Each token in Sequence A attends to all tokens in Sequence B
    Attention Matrix: 3×4 (queries × keys)
'''
class CrossAttention:
    def __init__(self, d_in_q, d_in_kv, d_out, dropout, num_heads, qkv_bias=False):
        self.d_out = d_out
        self.d_in_q = d_in_q
        self.d_in_kv = d_in_kv
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # Separate projections for different input sequences
        self.query = Linear(d_in_q, d_out, bias=qkv_bias)
        self.key = Linear(d_in_kv, d_out, bias=qkv_bias)
        self.value = Linear(d_in_kv, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)

    def forward(self, query_seq, key_value_seq):
        b, q_tokens, q_dim = query_seq.shape
        _, kv_tokens, kv_dim = key_value_seq.shape
        
        # Generate Q from query sequence, K,V from key-value sequence
        Q = self.query.forward(query_seq)
        K = self.key.forward(key_value_seq)
        V = self.value.forward(key_value_seq)
        
        # Reshape for multi-head attention
        Q = reshape(Q, (b, q_tokens, self.num_heads, self.head_dim))
        K = reshape(K, (b, kv_tokens, self.num_heads, self.head_dim))
        V = reshape(V, (b, kv_tokens, self.num_heads, self.head_dim))
        
        # Cross-attention: queries attend to key-value sequence
        att_scores = mat_mul_nd(Q, transpose_nd(K, 2, 3)) / np.sqrt(self.head_dim)
        attn_weights = softmax_nd(att_scores)
        context = mat_mul_nd(attn_weights, V)
        
        # Combine heads and project
        combined = combine_heads(context)
        return self.out_proj.forward(combined)
    
'''
Flash Attention optimizes memory usage and speed by processing attention in blocks instead of 
computing the full attention matrix at once. It uses online softmax to maintain numerical 
stability while dramatically reducing memory requirements from O(n²) to O(n).

    Standard Attention: Full Matrix    Flash Attention: Block Processing
    ┌─────────────────────────────┐    ┌───────┬───────┬───────┬───────┐
    │ Compute entire n×n matrix   │    │Block1 │Block2 │Block3 │Block4 │
    │ Memory: O(n²)               │    │ 64×64 │ 64×64 │ 64×64 │ 64×64 │
    │ ████████████████████████████│    │ ████  │ ████  │ ████  │ ████  │
    │ ████████████████████████████│    └───────┴───────┴───────┴───────┘
    │ ████████████████████████████│    Process one block at a time
    └─────────────────────────────┘    Memory: O(n)

Key benefits:
- Memory efficient: Processes sequences in small blocks (e.g., 64 tokens)
- Faster training: Reduces memory bandwidth bottlenecks
- Exact computation: Mathematically equivalent to standard attention
- Enables longer sequences: Can handle much larger context lengths
'''
class FlashAttention:
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False, block_size=64):
        self.d_out = d_out
        self.d_in = d_in
        self.dropout = dropout
        self.block_size = block_size
        
        # Same Q, K, V projections as standard attention
        self.query = Linear(d_in, d_out, bias=qkv_bias)
        self.key = Linear(d_in, d_out, bias=qkv_bias)
        self.value = Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = Linear(d_out, d_out)

    def forward(self, x):
        b, num_tokens, embed_dim = x.shape
        
        # Generate Q, K, V exactly like standard attention
        Q = self.query.forward(x)
        K = self.key.forward(x) 
        V = self.value.forward(x)
        
        # Flash attention: process in blocks instead of full matrix
        output = np.zeros_like(Q)
        scale = 1.0 / np.sqrt(self.d_out)
        
        for i in range(0, num_tokens, self.block_size):
            q_block = Q[:, i:i+self.block_size]
            output_block = np.zeros_like(q_block)
            
            # Online softmax state
            row_max = np.full((b, q_block.shape[1]), -np.inf)
            row_sum = np.zeros((b, q_block.shape[1]))
            
            for j in range(0, num_tokens, self.block_size):
                k_block = K[:, j:j+self.block_size]
                v_block = V[:, j:j+self.block_size]
                
                # Compute attention scores for this block
                scores = mat_mul(q_block, transpose(k_block)) * scale
                
                # Apply causal mask if needed
                if j + self.block_size > i:
                    for bi in range(scores.shape[0]):
                        for qi in range(scores.shape[1]):
                            for ki in range(scores.shape[2]):
                                if j + ki > i + qi:
                                    scores[bi, qi, ki] = -np.inf
                
                # Online softmax update
                block_max = np.max(scores, axis=2, keepdims=True)
                new_max = np.maximum(row_max[:, :, None], block_max)
                
                # Update previous contributions
                exp_scores = np.exp(scores - new_max)
                exp_prev = np.exp(row_max[:, :, None] - new_max)
                
                # Update running statistics
                row_sum = row_sum[:, :, None] * exp_prev + np.sum(exp_scores, axis=2, keepdims=True)
                row_max = new_max.squeeze(2)
                
                # Accumulate weighted values
                weights = exp_scores / row_sum
                output_block += mat_mul(weights, v_block)
            
            output[:, i:i+self.block_size] = output_block
        
        return self.out_proj.forward(output)
