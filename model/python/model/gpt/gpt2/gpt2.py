import numpy as np
from model.python.model.attention.attention import CausalAttention, ScaledDotProductAttention, MultiHeadAttention
from linear import Linear
from norm import LayerNorm
from model.python.model.feedforward import FeedForwardGPT
from model.embedding import Embedding

class Block():
    def __init__(self, cfg):
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForwardGPT(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        #self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1.forward(x)
        x = self.att.forward(x)   # Shape [batch_size, num_tokens, emb_size]
        #x = self.drop_shortcut(x)
        #x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2.forward(x)
        x = self.ff.forward(x)
        #x = self.drop_shortcut(x)
        #x = x + shortcut  # Add the original input back

        return x

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class GPT2Model:
    def __init__(self, cfg):
        self.tok_emb = Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = Embedding(cfg["context_length"], cfg["emb_dim"])
        self.blocks = [Block(cfg) for _ in range(cfg["n_layers"])]
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb.forward(in_idx)
        pos_embeds = self.pos_emb.forward(np.arange(seq_len))
        x = tok_embeds + pos_embeds
        for block in self.blocks:
            x = block.forward(x)
        x = self.final_norm.forward(x)
        logits = self.out_head.forward(x)
        return logits

'''

np.random.seed(42)
emd_dim = 512
num_heads = 5
x = np.random.randn(1, 3, emd_dim)  # (batch_size=1, num_tokens=3, emb_dim=5)

attention = CausalAttention(emd_dim, emd_dim, 5, 2)
res = attention.forward(x)
print('CausalAttention')
print(res)

attention = ScaledDotProductAttention(emd_dim, emd_dim, 5, 2)
res = attention.forward(x)
print('ScaledDotProductAttention')
print(res)

attention = MultiHeadAttention(emd_dim, emd_dim, 5, 2, num_heads)
res = attention.forward(x)
print('MultiHeadAttention')
print(res)'''

### Step 1: tokenzier

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(np.array(tokenizer.encode(txt1)))
batch.append(np.array(tokenizer.encode(txt2)))
batch = np.stack(batch, axis=0)
print(batch)

### Step 1: tokenzier

### Step 2: initialize a model

np.random.seed(123)
model = GPT2Model(GPT_CONFIG_124M)
logits = model.forward(batch)
print(f'Output shape {logits.shape}')
print(logits)

### Step 2: initialize a model

### MAIN -------