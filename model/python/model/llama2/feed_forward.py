from mtrx.module import Module
from mtrx.linear import Linear
import math

class SwiGLU:
    def __init__(self, d_in, d_hidden):
        self.w_gate = Linear(d_in, d_hidden)
        self.w_up = Linear(d_in, d_hidden) 
        self.w_down = Linear(d_hidden, d_in)
    
    def forward(self, x):
        gate = self.w_gate.forward(x)
        up = self.w_up.forward(x)
        # Swish activation: x * sigmoid(x) - element-wise for 1D arrays
        swish_gate = [g * (1 / (1 + math.exp(-g))) for g in gate]
        # Element-wise multiply (gating)
        gated = [swish_gate[i] * up[i] for i in range(len(up))]
        return self.w_down.forward(gated)

class FeedForwardLlama(Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Linear(cfg["emb_dim"], cfg["hidden_dim"])
        self.fc2 = Linear(cfg["emb_dim"], cfg["hidden_dim"])
        self.fc3 = Linear(cfg["hidden_dim"], cfg["emb_dim"])
        self.silu = SwiGLU(cfg["emb_dim"], cfg["hidden_dim"])

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
    