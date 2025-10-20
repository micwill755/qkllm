
class Block:
    def __init__(self):
        

class Llama2Model:
    def __init__(self, cfg):
        self.tok_emb = Embedding(cfg["vocab_size"], cfg["emb_dim"])
        pass

    def __call__(self, x):
        self.forward(x)

    def forward(self, x):
        pass