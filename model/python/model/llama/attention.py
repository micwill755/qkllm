
from mtrx.linear import Linear
from mtrx.tensor import Tensor   
from mtrx.mtrx import ones, reshape, mat_mul, mask, softmax

from rope import RoPE

import math

class MultiHeadAttention:
    def __init__(self, emb_dim, num_heads, use_rope):
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads
        self.use_rope = use_rope

        if use_rope:
            self.rope = RoPE(self.head_dim)
        
        self.k_W = Linear(emb_dim, emb_dim)
        self.q_W = Linear(emb_dim, emb_dim)
        self.v_W = Linear(emb_dim, emb_dim)
        self.out_proj = Linear(emb_dim, emb_dim)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        b, seq_len, dim = x.shape

        keys = self.k_W(x) # how am I accessed
        queries = self.q_W(x) # what I am looking for
        values = self.v_W(x) # what do I provide

        keys = reshape(keys, (b, seq_len, self.num_heads, self.head_dim))
        queries = reshape(queries, (b, seq_len, self.num_heads, self.head_dim))
        values = reshape(values, (b, seq_len, self.num_heads, self.head_dim))

        # [b, seq_len, heads, head_dim] -> [b, heads, seq_len, head_dim]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.use_rope:
            keys = self.rope(keys)
            queries = self.rope(queries)

        # queries @ keys.T (2, 3) = [seq_len, head_dim] -> [seq_len, head_dim]
        # TODO - write c kernels to speed up tensor operations and remove recreating tensors
        attn_scores = mat_mul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_scores = mask(attn_scores)
        attn_weights = softmax(attn_scores)

        context = mat_mul(attn_weights, values)
        context = context.transpose(1, 2)
        context = reshape(context, (b, seq_len, self.emb_dim))

        return self.out_proj(context)
 
class GroupQueryAttention:
    def __init__(self, emb_dim, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = emb_dim // num_heads

        self.q_W = Linear(emb_dim, emb_dim)
        self.k_W = Linear(emb_dim, num_kv_heads * self.head_dim)
        self.v_W = Linear(emb_dim, num_kv_heads * self.head_dim)
        self.out_proj = Linear(emb_dim, emb_dim)
    
    def __call__(self, x):
        return self.forward(x)
    
    def _repeat_kv_heads(self, x, num_groups):
        # x shape: [b, num_kv_heads, seq_len, head_dim]
        # output: [b, num_kv_heads * num_groups, seq_len, head_dim]
        b, num_kv_heads, seq_len, head_dim = x.shape
        result = Tensor((b, num_kv_heads * num_groups, seq_len, head_dim))
        
        for batch_idx in range(b):
            for kv_head_idx in range(num_kv_heads):
                for group_idx in range(num_groups):
                    output_head_idx = kv_head_idx * num_groups + group_idx
                    for seq_idx in range(seq_len):
                        for dim_idx in range(head_dim):
                            result[batch_idx][output_head_idx][seq_idx][dim_idx] = x[batch_idx][kv_head_idx][seq_idx][dim_idx]
        
        return result

    def forward(self, x):
        b, seq_len, emb_dim = x.shape
        
        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        queries = reshape(query, (b, seq_len, self.num_heads, self.head_dim))
        keys = reshape(key, (b, seq_len, self.num_kv_heads, self.head_dim))
        values = reshape(value, (b, seq_len, self.num_kv_heads, self.head_dim))

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Repeat K/V heads to match Q heads
        keys = self._repeat_kv_heads(keys, (self.num_heads // self.num_kv_heads))
        values = self._repeat_kv_heads(values, (self.num_heads // self.num_kv_heads))

        attn_scores = mat_mul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_scores = mask(attn_scores)
        attn_weights = softmax(attn_scores)
        context = mat_mul(attn_weights, values)
        context = context.transpose(1, 2)

        context = reshape(context, (b, seq_len, emb_dim))

        return self.out_proj(context)