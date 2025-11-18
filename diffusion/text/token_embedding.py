import random

class TokenEmbedding:
    def __init__(self, vocab_size, emb_dim):
        """
        Token embedding layer
        
        Args:
            vocab_size: Number of tokens in vocabulary (e.g., 50257 for GPT-2)
            emb_dim: Dimension of embedding vectors (e.g., 128)
        """
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        
        # Initialize embedding matrix with small random values
        # Shape: (vocab_size, emb_dim)
        self.weights = [[random.gauss(0, 0.02) for _ in range(emb_dim)] 
                       for _ in range(vocab_size)]
    
    def forward(self, tokens):
        """
        Convert token IDs to embeddings
        
        Args:
            tokens: List of token IDs, e.g., [464, 3797, 3332]
        
        Returns:
            List of embedding vectors, shape (len(tokens), emb_dim)
        """
        embeddings = []
        for token_id in tokens:
            embeddings.append(self.weights[token_id])
        return embeddings
    
    def __call__(self, tokens):
        return self.forward(tokens)
