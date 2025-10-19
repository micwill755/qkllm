import torch
import torch.nn as nn
import math

# Q1 Code multihead attention from scratch

class Attention(nn.Module):
    def __init__(self, d_in, d_out, context_length):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        self.query = nn.Linear(d_in, d_out)
        self.key = nn.Linear(d_in, d_out)
        self.value = nn.Linear(d_in, d_out)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        att_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_out)
        att_scores = att_scores.masked_fill(self.mask.bool(), float('-inf'))
        attn_weights = torch.softmax(att_scores, dim=-1)
        context = attn_weights @ v
        return context

attention = Attention(d_in=4, d_out=8, context_length=3)
x = torch.randn(2, 3, 4) # batch, sequence length, d_in 4
output = attention(x)
print(output)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.query = nn.Linear(d_in, d_out)
        self.key = nn.Linear(d_in, d_out)
        self.value = nn.Linear(d_in, d_out)

        self.out_proj = nn.Linear(d_in, d_out)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch, tokens, emd_dim = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # reshape to head dimension
        queries = q.view(batch, tokens, self.num_heads, self.head_dim)
        keys = k.view(batch, tokens, self.num_heads, self.head_dim)
        values = v.view(batch, tokens, self.num_heads, self.head_dim)

        # transpose from [b, seq, heads, head_dim] -> [b, heads, seq, dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        att_scores = queries @ keys.transpose(2, 3) / math.sqrt(self.head_dim)
        att_scores = att_scores.masked_fill(self.mask.bool(), -torch.inf)

        attn_weights = torch.softmax(att_scores, dim=-1)
        context = attn_weights @ values
        # transpose back from [b, heads, seq, dim] -> [b, seq, heads, head_dim]
        context = context.transpose(1, 2)
        context = context.reshape(batch, tokens, self.d_out)

        return self.out_proj(context)

m_attention = MultiHeadAttention(128, 128, 16, 4)
x = torch.randn(1, 16, 128)
output = m_attention(x)
print(output)