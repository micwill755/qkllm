import torch.nn as nn
from .attention import MultiHeadLatentAttention
from .moe import MoE
from .norms import RMSNorm


class DeepSeekBlock(nn.Module):
    """DeepSeek transformer block with latent attention and MoE"""
    
    def __init__(self, emb_dim, num_heads, seq_len, num_experts, top_k, expert_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len

        # Layer components
        self.rms_norm1 = RMSNorm(emb_dim)
        self.attention = MultiHeadLatentAttention(emb_dim, num_heads, seq_len)
        self.rms_norm2 = RMSNorm(emb_dim)
        self.moe = MoE(emb_dim, num_experts, top_k, expert_dim)

    def forward(self, x):
        # First sub-layer: Attention with residual connection
        attn_output = self.attention(self.rms_norm1(x))
        x = x + attn_output
        
        # Second sub-layer: MoE with residual connection
        moe_output = self.moe(self.rms_norm2(x))
        x = x + moe_output
        
        return x