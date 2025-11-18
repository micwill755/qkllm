from time_embedding import TimeEmbedding
from token_embedding import TokenEmbedding
from mx.linear import Linear
from mx.mtrx import mat_mul, reshape
from mx.norm import LayerNorm

import mx

# start with simple MHA
class MultiHeadAttention:
    def __init__(self, emb_dim, num_heads):
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads
        
        self.q_W = Linear(emb_dim, emb_dim)
        self.k_W = Linear(emb_dim, emb_dim)
        self.v_W = Linear(emb_dim, emb_dim)

        self.out_proj = Linear(emb_dim, emb_dim)
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        # TODO: add underlying functions to calculate 
        # using 1d array so we dont need to transpose

# start with using simple blocks from GPT2
class Block (mx.Module):
    def __init__(self, emb_dim, n_heads):
        self.att = MultiHeadAttention(emb_dim, num_heads=n_heads)
        #self.ff = FeedForwardGPT(cfg)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # TODO
        
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1.forward(x)
        x = self.att.forward(x)   # Shape [batch_size, num_tokens, emb_size]
        #x = self.drop_shortcut(x)
        #x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2.forward(x)
        #x = self.ff.forward(x)
        #x = self.drop_shortcut(x)
        #x = x + shortcut  # Add the original input back

        return x

class DiffusionModel(mx.Module):
    def __init__(self, vocab_size, emb_dim, n_heads, n_layers):
        self.token_embedding = TokenEmbedding(vocab_size, emb_dim)
        self.time_embedding = TimeEmbedding(emb_dim)
        self.blocks = [Block(emb_dim, n_heads) for _ in range(n_layers)]
        self.output_proj = Linear(emb_dim, emb_dim)

    def forward(self, tokens, timestep):
        token_emb = self.token_embedding(tokens)
        time_emb = self.time_embedding(timestep)
        combined = token_emb + time_emb
    
        # TODO