import random
import numpy as np

class Embedding:
    def __init__(self, vocab_size, emb_dim):
        self.weight = np.random.randn(vocab_size, emb_dim)
    
    def forward(self, idx):
        return self.weight[idx]
    
class Embedding1d:
    def __init__(self, vocab_size, emb_dim):
        self.weight = [random.gauss(0, 0.02) for _ in range (vocab_size * emb_dim)]
    def forward(self, idx):
        return self.weight[idx]