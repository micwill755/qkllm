import random
from mtrx.tensor import Tensor

class Embedding:
    def __init__(self, vocab_size, emb_dim):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.weight = []
        for _ in range(vocab_size):
            token_emb = [random.gauss(0, 0.02) for _ in range(emb_dim)]
            self.weight.append(token_emb)
    
    def __call__(self, tokens):
        return self.forward(tokens)

    def forward(self, tokens):
        batch, seq_len = tokens.shape
        result = Tensor((batch, seq_len, self.emb_dim))
        
        for b in range(batch):
            for t in range(seq_len):
                token_val = tokens.tensor[b][t]
                token_id = int(abs(token_val)) % self.vocab_size
                result[b][t] = self.weight[token_id]
        
        return result

