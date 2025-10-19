import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation function (simplified version for testing)"""
    
    def __init__(self, dim1, dim2):
        super().__init__()
        self.linear = nn.Linear(dim1, dim2)
        
    def forward(self, x):
        return torch.relu(self.linear(x))  # Simplified - should be SiLU with gating