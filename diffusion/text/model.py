import math

class TimeEmbedding:
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
    
    def forward(self, timestep):
        """Sinusoidal time embedding"""
        half_dim = self.emb_dim // 2
        emb = []
        for i in range(half_dim):
            freq = math.exp(-math.log(10000.0) * i / half_dim)
            emb.append(math.sin(timestep * freq))
            emb.append(math.cos(timestep * freq))
        return emb[:self.emb_dim]


class Embedding:
    def __init__(self, vocab_size, emb_dim):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        # Initialize random embeddings
        self.weights = [[random.random() * 0.02 - 0.01 for _ in range(emb_dim)] 
                       for _ in range(vocab_size)]
    
    def forward(self, tokens):
        return [self.weights[token] for token in tokens]


class Linear:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights = [[random.random() * 0.02 - 0.01 for _ in range(out_dim)] 
                       for _ in range(in_dim)]
        self.bias = [0.0] * out_dim
    
    def forward(self, x):
        # x: (seq_len, in_dim) -> (seq_len, out_dim)
        output = []
        for seq_vec in x:
            out_vec = self.bias.copy()
            for i, val in enumerate(seq_vec):
                for j in range(self.out_dim):
                    out_vec[j] += val * self.weights[i][j]
            output.append(out_vec)
        return output


class DiffusionTransformer:
    def __init__(self, vocab_size, emb_dim, num_layers=4):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.token_embedding = Embedding(vocab_size, emb_dim)
        self.time_embedding = TimeEmbedding(emb_dim)
        self.output_head = Linear(emb_dim, vocab_size)
    
    def forward(self, noisy_tokens, timestep):
        # Embed tokens
        x = self.token_embedding.forward(noisy_tokens)
        
        # Get time embedding and add to all positions
        t_emb = self.time_embedding.forward(timestep)
        for i in range(len(x)):
            for j in range(self.emb_dim):
                x[i][j] += t_emb[j]
        
        # Simple processing (in practice, use transformer blocks)
        # For minimal implementation, just pass through output head
        logits = self.output_head.forward(x)
        
        return logits


import random
