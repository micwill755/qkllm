from mx import Module, Linear, Tensor   
import math

class SiLU(Module):
    def __init__(self):
        super().__init__()
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # x * sigmoid(x) element-wise
        batch, seq_len, hidden = x.shape
        result = Tensor((batch, seq_len, hidden))
        
        for b in range(batch):
            for s in range(seq_len):
                for h in range(hidden):
                    val = x[b][s][h]
                    result[b][s][h] = val * (1 / (1 + math.exp(-val)))
        
        return result

class FeedForwardLlama(Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = Linear(cfg["emb_dim"], cfg["hidden_dim"])
        self.fc2 = Linear(cfg["emb_dim"], cfg["hidden_dim"])
        self.fc3 = Linear(cfg["hidden_dim"], cfg["emb_dim"])
        self.silu = SiLU()

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
    