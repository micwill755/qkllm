import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding import Embedding
from attention.attention import GroupedQueryAttention
from mtrx.mtrx import ones
from mtrx.tensor import Tensor   
from mtrx.module import Module
from feed_forward import FeedForwardLlama
from mtrx.linear import Linear

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
    
class RMSNorm (Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weights = ones((emb_dim, ))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        batch, seq_len, emb_dim = x.shape
        result = Tensor((batch, seq_len, emb_dim))

        for b in range(batch):
            for t in range(seq_len):
                ms = sum (x[b][t][e] ** 2 for e in range(emb_dim)) / emb_dim
                rms = math.sqrt(ms + self.eps)
                for e in range(emb_dim):
                    result[b][t][e] = (x[b][t][e] / rms) * self.weights[e]
        
        return result

class Block:
    def __init__(self, cfg):
        self.attn = GroupQueryAttention(cfg["emb_dim"], cfg["num_heads"], cfg["num_kv_heads"])
        self.feed_forward = FeedForwardLlama(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # Attention block with residual
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        
        # Feed-forward block with residual
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = shortcut + x
        
        return x

class Llama2Model:
    def __init__(self, cfg):
        self.tok_emb = Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.blocks = [Block(cfg) for _ in range(cfg["n_layers"])]
        self.final = RMSNorm(cfg["emb_dim"])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.tok_emb(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final(x)
        return x

LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008      # NEW: Size of the intermediate dimension in FeedForward
    #"dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}

LLAMA2_CONFIG_MINI = {
    "vocab_size": 100,       # Minimal vocabulary
    "context_length": 64,    # Short context
    "emb_dim": 128,          # Small embedding
    "n_heads": 4,            # Few heads
    "n_kv_heads": 4,            # Few heads
    "n_layers": 2,           # Few layers
    "hidden_dim": 512        # Small hidden dimension
}

model = Llama2Model(LLAMA2_CONFIG_MINI)

input = Tensor((1, 10, LLAMA2_CONFIG_MINI["emb_dim"]), use_rand=True)
output = model(input)
print(output)