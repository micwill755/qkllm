from mtrx.module import Module
from mtrx.linear import Linear
from mtrx.tensor import Tensor   

import math

class SwiGLU(Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.w_gate = Linear(d_in, d_hidden)
        self.w_up = Linear(d_in, d_hidden) 
        self.w_down = Linear(d_hidden, d_in)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        gate = self.w_gate.forward(x)
        up = self.w_up.forward(x)
        
        # Apply SiLU: x * sigmoid(x) element-wise, then multiply with up
        batch, seq_len, hidden = gate.shape
        result = Tensor((batch, seq_len, hidden))
        
        for b in range(batch):
            for s in range(seq_len):
                for h in range(hidden):
                    g = gate[b][s][h]
                    silu = g * (1 / (1 + math.exp(-g)))  # SiLU activation
                    result[b][s][h] = silu * up[b][s][h]
        
        return self.w_down.forward(result)

class FeedForwardLlama(Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Linear(cfg["emb_dim"], cfg["hidden_dim"])
        self.fc2 = Linear(cfg["emb_dim"], cfg["hidden_dim"])
        self.fc3 = Linear(cfg["hidden_dim"], cfg["emb_dim"])
        self.silu = SwiGLU(cfg["emb_dim"], cfg["hidden_dim"])

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
    