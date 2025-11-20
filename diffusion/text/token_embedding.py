import mx

class TokenEmbedding:
    def __init__(self, vocab_size, emb_dim):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        # Use mx.Tensor instead of Python lists
        self.weights = mx.randn([vocab_size, emb_dim])
    
    def forward(self, tokens):
        # tokens: mx.Tensor of shape (batch, seq_len) with integer indices
        # Use tensor indexing instead of Python loops
        return self.weights[tokens]  # Returns (batch, seq_len, emb_dim)
    
    def __call__(self, tokens):
        return self.forward(tokens)