import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from matrix_helper import transpose, mat_mul

class LoRALinear:
    """
    LoRA-enabled Linear layer that adds low-rank adaptation to existing linear layers.
    
    LoRA decomposes weight updates as: ΔW = A @ B
    Where A is (d_out, rank) and B is (rank, d_in)
    Final output: y = (W + α/r * A @ B) @ x + b
    """
    
    def __init__(self, d_in, d_out, bias=True, lora_rank=16, lora_alpha=16, lora_dropout=0.1):
        # Original linear layer weights (frozen during LoRA training)
        self.weight = np.random.randn(d_out, d_in) * 0.02
        self.bias = np.random.randn(d_out) * 0.02 if bias else None
        
        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.scaling = lora_alpha / lora_rank
        
        # LoRA matrices - A initialized with random, B initialized to zero
        self.lora_A = np.random.randn(d_out, lora_rank) * 0.01
        self.lora_B = np.zeros((lora_rank, d_in))
        
        # Training flags
        self.lora_enabled = True
        self.freeze_original = True
    
    def forward(self, x):
        """Forward pass with optional LoRA adaptation"""
        # Original linear transformation
        if len(x.shape) == 2:  # 2D input
            out = mat_mul(x, transpose(self.weight))
        elif len(x.shape) == 3:  # 3D input (batch, tokens, emb_dim)
            batch_size, num_tokens, emb_dim = x.shape
            d_out = self.weight.shape[0]
            out = np.zeros((batch_size, num_tokens, d_out))
            weight_t = transpose(self.weight)
            
            for batch in range(batch_size):
                out[batch] = mat_mul(x[batch], weight_t)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Add LoRA adaptation if enabled
        if self.lora_enabled:
            lora_out = self._lora_forward(x)
            out = out + self.scaling * lora_out
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def _lora_forward(self, x):
        """LoRA forward pass: x @ B^T @ A^T"""
        if len(x.shape) == 2:
            # x @ B^T @ A^T = x @ (A @ B)^T
            lora_weight = mat_mul(self.lora_A, self.lora_B)  # (d_out, d_in)
            return mat_mul(x, transpose(lora_weight))
        elif len(x.shape) == 3:
            batch_size, num_tokens, emb_dim = x.shape
            d_out = self.lora_A.shape[0]
            lora_out = np.zeros((batch_size, num_tokens, d_out))
            
            lora_weight = mat_mul(self.lora_A, self.lora_B)
            lora_weight_t = transpose(lora_weight)
            
            for batch in range(batch_size):
                lora_out[batch] = mat_mul(x[batch], lora_weight_t)
            
            return lora_out
    
    def enable_lora(self):
        """Enable LoRA adaptation"""
        self.lora_enabled = True
    
    def disable_lora(self):
        """Disable LoRA adaptation (use original weights only)"""
        self.lora_enabled = False
    
    def merge_lora(self):
        """Merge LoRA weights into original weights permanently"""
        if self.lora_enabled:
            lora_weight = mat_mul(self.lora_A, self.lora_B)
            self.weight = self.weight + self.scaling * lora_weight
            # Reset LoRA matrices
            self.lora_A = np.zeros_like(self.lora_A)
            self.lora_B = np.zeros_like(self.lora_B)
    
    def get_lora_parameters(self):
        """Get LoRA parameters for training"""
        return {'lora_A': self.lora_A, 'lora_B': self.lora_B}
    
    def get_trainable_params_count(self):
        """Count trainable parameters"""
        if self.freeze_original:
            return self.lora_A.size + self.lora_B.size
        else:
            total = self.weight.size + self.lora_A.size + self.lora_B.size
            if self.bias is not None:
                total += self.bias.size
            return total