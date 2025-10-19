import math
import torch
import torch.nn as nn


class ROPE(nn.Module):
    """Rotary Position Embedding"""
    
    def __init__(self, head_dim, max_seq_length=2048, base=10000):
        super().__init__()
        i = torch.arange(head_dim // 2, dtype=torch.float32)
        self.frequencies = 1 / (base ** (2 * i / head_dim))
        self.positions = torch.arange(max_seq_length, dtype=torch.float32)
        self.angles = torch.outer(self.positions, self.frequencies)

    def forward(self, x):
        # x shape: (batch, num_heads, seq_len, head_dim)
        batch, num_heads, seq_len, head_dim = x.shape
        x = x.view(batch, num_heads, seq_len, head_dim // 2, 2)
        
        for pos in range(seq_len):
            for pair_idx in range(head_dim // 2):
                angle = self.angles[pos, pair_idx]
                X = x[:, :, pos, pair_idx, 0]
                Y = x[:, :, pos, pair_idx, 1]
                x[:, :, pos, pair_idx, 0] = X * torch.cos(angle) - Y * torch.sin(angle)
                x[:, :, pos, pair_idx, 1] = X * torch.sin(angle) + Y * torch.cos(angle)
        
        return x.view(batch, num_heads, seq_len, head_dim)


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention with cross-attention to compressed latent tokens"""
    
    def __init__(self, emb_dim, num_heads, max_seq_length, latent_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.latent_dim = latent_dim

        self.rope = ROPE(head_dim=self.head_dim, max_seq_length=max_seq_length)

        self.latent_tokens = nn.Linear(max_seq_length * emb_dim, latent_dim * emb_dim)
        self.query_W = nn.Linear(emb_dim, emb_dim)
        self.key_W = nn.Linear(emb_dim, emb_dim)
        self.value_W = nn.Linear(emb_dim, emb_dim)
        self.output_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        
        # Compress sequence to latent tokens
        x_flat = x.view(batch_size, -1)  # (batch, seq_len * emb_dim)
        latent_compressed = self.latent_tokens(x_flat)  # (batch, latent_dim * emb_dim)
        latent = latent_compressed.view(batch_size, self.latent_dim, emb_dim)

        # Generate Q, K, V (cross-attention)
        query = self.query_W(x)        # (batch, seq_len, emb_dim)
        key = self.key_W(latent)       # (batch, latent_dim, emb_dim)
        value = self.value_W(latent)   # (batch, latent_dim, emb_dim)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, self.latent_dim, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, self.latent_dim, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to queries only
        query_r = self.rope(query)
        key_r = key  # No RoPE for latent tokens
            
        # Attention computation
        attn_scores = query_r @ key_r.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ value  # (batch, num_heads, seq_len, head_dim)

        # Combine heads back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        
        return self.output_proj(attn_output)