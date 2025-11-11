import math
import os
import sys
from pathlib import Path

from embedding import Embedding
from mtrx.mtrx import ones, reshape, mat_mul, mask, softmax
from mtrx.tensor import Tensor   
from mtrx.module import Module
from feed_forward import FeedForwardLlama
from mtrx.linear import Linear

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

class RMSNorm (Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weights = ones((emb_dim, ))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        batch, seq_len, emb_dim = x.shape
        result = Tensor((batch, seq_len, emb_dim))

        for b in range(batch):
            for t in range(seq_len):
                ms = sum (x[b][t][e] ** 2 for e in range(emb_dim)) / emb_dim
                rms = math.sqrt(ms + self.eps)
                for e in range(emb_dim):
                    result[b][t][e] = (x[b][t][e] / rms) * self.weights[e]
        
        return result

class Block:
    def __init__(self, cfg):
        self.attn = GroupQueryAttention(cfg["emb_dim"], cfg["n_heads"], cfg["n_kv_heads"])
        self.feed_forward = FeedForwardLlama(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # Attention block with residual
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        
        # Feed-forward block with residual
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = shortcut + x
        
        return x

class Llama2Model:
    def __init__(self, cfg):
        self.tok_emb = Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.blocks = [Block(cfg) for _ in range(cfg["n_layers"])]
        self.final = RMSNorm(cfg["emb_dim"])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.tok_emb(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final(x)
        return x

LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008      # NEW: Size of the intermediate dimension in FeedForward
    #"dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}

LLAMA2_CONFIG_MINI = {
    "vocab_size": 100,       # Minimal vocabulary
    "context_length": 64,    # Short context
    "emb_dim": 128,          # Small embedding
    "n_heads": 4,            # Few heads
    "n_kv_heads": 4,            # Few heads
    "n_layers": 2,           # Few layers
    "hidden_dim": 512        # Small hidden dimension
}

model = Llama2Model(LLAMA2_CONFIG_MINI)

input = Tensor((1, 10, LLAMA2_CONFIG_MINI["emb_dim"]), use_rand=True)
output = model(input)
print(output)