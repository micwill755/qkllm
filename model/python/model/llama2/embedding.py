import random

class Embedding:
    def __init__(self, vocab_size, emb_dim):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        # Initialize random weights for each token
        self.weight = []
        for _ in range(vocab_size):
            token_emb = [random.gauss(0, 0.02) for _ in range(emb_dim)]
            self.weight.append(token_emb)
    
    def __call__(self, tokens):
        return self.forward(tokens)

    def forward(self, tokens):
        """Convert list of token IDs to embeddings"""
        embeddings = []
        for token_id in tokens:
            embeddings.append(self.weight[token_id])
        return embeddings