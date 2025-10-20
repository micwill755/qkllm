from .embedding import Embedding

class Block:
    def __init__(self):
        pass

class Llama2Model:
    def __init__(self, cfg):
        self.tok_emb = Embedding(cfg["vocab_size"], cfg["emb_dim"])
        pass

    def __call__(self, x):
        self.forward(x)

    def forward(self, x):
        pass


LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008      # NEW: Size of the intermediate dimension in FeedForward
    #"dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}

model = Llama2Model(LLAMA2_CONFIG_7B)