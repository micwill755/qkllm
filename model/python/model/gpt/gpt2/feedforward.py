from model.linear import Linear, Linear1d
from model.activation_functions import GELU, SwiGLU
from qmx.module import Module

class FeedForwardGPT(Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear1 = Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        self.gelu = GELU()
        self.linear2 = Linear(4 * cfg["emb_dim"], cfg["emb_dim"])

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.gelu.forward(x)
        x = self.linear2.forward(x)
        return x
    
class FeedForwardLlama(Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
        self.silu = SwiGLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
    
'''
    Each expert is a simple Feed-Forward Network (FFN) that processes tokens independently.
    Expert Architecture:
    Input [emb_dim] 
        ↓
    Linear1: [emb_dim → expert_dim] 
        ↓
    Activation (SwiGLU or ReLU)
        ↓  
    Linear2: [expert_dim → emb_dim]
        ↓
    Output [emb_dim]
'''
class Expert():
    def __init__(self, emb_dim, expert_dim):
        self.layer1 = Linear1d(emb_dim, expert_dim)
        self.swiGLU = SwiGLU(expert_dim, expert_dim) 
        self.layer2 = Linear1d(expert_dim, emb_dim)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.swiGLU.forward(x)
        x = self.layer2.forward(x)

        return x