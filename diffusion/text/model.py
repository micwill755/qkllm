from time_embedding import TimeEmbedding
from token_embedding import TokenEmbedding
from mx.linear import Linear
from mx import LayerNorm, RMSNorm, Tensor

import mx

# start with simple MHA
class MultiHeadAttention:
    def __init__(self, emb_dim, num_heads):
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads
        self.num_heads = num_heads

        self.q_W = Linear(emb_dim, emb_dim)
        self.k_W = Linear(emb_dim, emb_dim)
        self.v_W = Linear(emb_dim, emb_dim)

        self.out_proj = Linear(emb_dim, emb_dim)
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        b, num_tokens, emb_dim = x.shape

        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        queries = query.reshape([b, num_tokens, self.num_heads, self.head_dim])
        keys = key.reshape([b, num_tokens, self.num_heads, self.head_dim])
        values = value.reshape([b, num_tokens, self.num_heads, self.head_dim])

        # [b, num_tokens, num_heads, head_dim] -> [b, num_heads, num_tokens, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) / (self.head_dim ** 0.5)
        attn_weights = mx.softmax(attn_scores, dim=-1)
        context = attn_weights @ values  # -> [b, num_heads, num_tokens, head_dim]

        context = context.transpose(1, 2)  # -> [b, num_tokens, num_heads, head_dim]
        context = context.reshape([b, num_tokens, self.emb_dim])
        
        return self.out_proj(context)

class FeedForward(mx.Module):
    def __init__(self, emb_dim):
        self.fc1 = Linear(emb_dim, 4 * emb_dim)
        self.fc2 = Linear(4 * emb_dim, emb_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = mx.gelu(x) 
        x = self.fc2(x)
        return x
    
# start with using simple blocks from GPT2
class Block (mx.Module):
    def __init__(self, emb_dim, n_heads):
        self.att = MultiHeadAttention(emb_dim, num_heads=n_heads)
        self.ff = FeedForward(emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1.forward(x)
        x = self.att.forward(x)   # Shape [batch_size, num_tokens, emb_size]
        #x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2.forward(x)
        x = self.ff.forward(x)
        #x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

class DiffusionModel(mx.Module):
    def __init__(self, vocab_size, emb_dim, n_heads, n_layers):
        self.token_embedding = TokenEmbedding(vocab_size, emb_dim)
        self.time_embedding = TimeEmbedding(emb_dim)
        self.blocks = [Block(emb_dim, n_heads) for _ in range(n_layers)]
        self.output_proj = Linear(emb_dim, emb_dim)

    def forward(self, tokens, timestep):
        token_emb = self.token_embedding(tokens)  # [batch, seq_len, emb_dim]
        time_emb = self.time_embedding(timestep)  # [batch, emb_dim]
        time_emb = time_emb.reshape([time_emb.shape[0], 1, time_emb.shape[1]])  # [batch, 1, emb_dim]
        x = token_emb + time_emb  # Broadcasting works now
    
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to output
        x = self.output_proj(x)
        
        return x
    
# Test parameters
vocab_size = 1000
emb_dim = 128
n_heads = 4
n_layers = 2
batch_size = 2
seq_len = 10

# Initialize model
model = DiffusionModel(vocab_size, emb_dim, n_heads, n_layers)

# Create dummy inputs with INTEGER token IDs (not random floats!)
tokens = mx.zeros([batch_size, seq_len])  # All zeros for now (valid token IDs)
timestep = mx.randn([batch_size, 1])

# Forward pass
output = model.forward(tokens, timestep)

# Verify output shape
print(f"Input shape: {tokens.shape}")
print(f"Timestep shape: {timestep.shape}")
print(f"Output shape: {output.shape}")
assert output.shape == (batch_size, seq_len, emb_dim), "Output shape mismatch!"

print("✓ Model test passed!")
