import numpy as np
from .lib.matrix_helper import transpose, mat_mul, reshape

import random

# temp
class Linear:
    def __init__(self, d_in, d_out, bias=True):
        self.weight = np.random.randn(d_out, d_in)
        self.bias = np.random.randn(d_out) if bias else None
    
    # y = mx + b
    def forward(self, x):
        # TEMP
        if len(x.shape) == 2:  # 2D input (tokens, emb_dim)
            out = mat_mul(x, transpose(self.weight))
            if self.bias is not None:
                out = out + self.bias
            return out
        elif len(x.shape) == 3:  # 3D input (batch, tokens, emb_dim)
            batch_size, num_tokens, emb_dim = x.shape
            d_out = self.weight.shape[0]
            out = np.zeros((batch_size, num_tokens, d_out))
            
            weight_t = transpose(self.weight)
            
            for batch in range(batch_size):
                out[batch] = mat_mul(x[batch], weight_t)
                if self.bias is not None:
                    out[batch] = out[batch] + self.bias
            
            return out
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

class Linear1d:
    # TODO: NEED TO UNDER STAND THIS IMPLEMENTATION
    def __init__(self, d_in, d_out, bias=True):
        self.d_in = d_in
        self.d_out = d_out
        self.weight = [random.gauss(0, 0.02) for _ in range (d_out * d_in)]  # [d_out, d_in] - smaller init
        self.bias = [random.gauss(0, 0.01) for _ in range (d_out)] if bias else None
    
    # y = mx + b
    def forward(self, x):
        # Calculate number of input rows (batch * seq_len)
        input_rows = len(x) // self.d_in
        
        # Create output array
        out = [0.0 for _ in range(input_rows * self.d_out)]
        
        # Matrix multiplication: x @ weight.T
        for row in range(input_rows):
            for col in range(self.d_out):
                sum_val = 0.0
                for k in range(self.d_in):
                    x_idx = row * self.d_in + k
                    w_idx = col * self.d_in + k  # weight is [d_out, d_in], so col-th row starts at col*d_in
                    sum_val += x[x_idx] * self.weight[w_idx]
                
                out_idx = row * self.d_out + col
                out[out_idx] = sum_val
                
                # Add bias if it exists
                if self.bias is not None:
                    out[out_idx] += self.bias[col]
        
        return out