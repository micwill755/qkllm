import math
import os
import sys
from pathlib import Path
import mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding import Embedding

from attention import MultiHeadAttention
from feed_forward import FeedForwardLlama

import tiktoken


class RMSNorm(mx.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weights = [1.0] * emb_dim

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        batch, seq_len, emb_dim = x.shape
        result = mx.Tensor((batch, seq_len, emb_dim))

        for b in range(batch):
            for t in range(seq_len):
                ms = sum(x[b][t][e] ** 2 for e in range(emb_dim)) / emb_dim
                rms = math.sqrt(ms + self.eps)
                for e in range(emb_dim):
                    result[b][t][e] = (x[b][t][e] / rms) * self.weights[e]
        
        return result

class Block:
    def __init__(self, cfg):
        self.attn = MultiHeadAttention(cfg["emb_dim"], cfg["n_heads"], use_rope=True)
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

def test_llama2_output():
    model = Llama2Model(LLAMA2_CONFIG_MINI)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Test input
    text = "Hello world"
    tokens = tokenizer.encode(text)
    input = mx.Tensor([tokens])
    
    # Forward pass
    output = model(input)
    
    # Assertions
    batch, seq_len, emb_dim = output.shape
    assert batch == 1, f"Expected batch=1, got {batch}"
    assert seq_len == len(tokens), f"Expected seq_len={len(tokens)}, got {seq_len}"
    assert emb_dim == LLAMA2_CONFIG_MINI["emb_dim"], f"Expected emb_dim={LLAMA2_CONFIG_MINI['emb_dim']}, got {emb_dim}"
    
    # Check output is not all zeros
    has_nonzero = False
    for b in range(batch):
        for s in range(seq_len):
            for e in range(emb_dim):
                if output[b][s][e] != 0:
                    has_nonzero = True
                    break
                
    assert has_nonzero, "Output is all zeros"
    
    print("✓ All tests passed")
    return output

# Run test
test_llama2_output()
