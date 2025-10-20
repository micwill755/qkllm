import math
from .tensor import Tensor
from .torch import mat_mul, randn

class Linear:
    def __init__(self, d_in, d_out):
        self.weights = randn((d_in, d_out))
        self.bias = 0
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # y = mx + b
        if len(x.shape) == 3:
            batch, seq_len, emb_dim = x.shape
            result = Tensor((batch, seq_len, emb_dim))
            for b in range(batch):
                result[b] = mat_mul(x[b], self.weights) #+ self.bias
            return result
        return 0