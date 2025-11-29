# vit.py - Vision Transformer from scratch

import torch
import torch.nn as nn
import torch.nn.functional as F

from patch_embedding import PatchEmbedding

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, n_tokens, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch, n_tokens, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: (Q @ K^T) / sqrt(d_k)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch, num_heads, n_tokens, n_tokens)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = attn @ v  # (batch, num_heads, n_tokens, head_dim)
        x = x.transpose(1, 2)  # (batch, n_tokens, num_heads, head_dim)
        x = x.reshape(batch_size, n_tokens, embed_dim)
        
        x = self.proj(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block: Attention + MLP with residual connections"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x):
        # Pre-norm architecture (used in ViT)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image classification"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification head (use only class token)
        x = self.norm(x)
        cls_token_final = x[:, 0]  # (batch, embed_dim)
        x = self.head(cls_token_final)
        
        return x


# Example usage and testing
if __name__ == "__main__":
    # Create ViT-Base model (similar to original paper)
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ViT variants
    print("\n--- ViT Model Variants ---")
    
    # ViT-Tiny
    vit_tiny = VisionTransformer(embed_dim=192, depth=12, num_heads=3)
    print(f"ViT-Tiny params: {sum(p.numel() for p in vit_tiny.parameters()):,}")
    
    # ViT-Small
    vit_small = VisionTransformer(embed_dim=384, depth=12, num_heads=6)
    print(f"ViT-Small params: {sum(p.numel() for p in vit_small.parameters()):,}")
    
    # ViT-Base (default)
    print(f"ViT-Base params: {sum(p.numel() for p in model.parameters()):,}")
    
    # ViT-Large
    vit_large = VisionTransformer(embed_dim=1024, depth=24, num_heads=16)
    print(f"ViT-Large params: {sum(p.numel() for p in vit_large.parameters()):,}")
