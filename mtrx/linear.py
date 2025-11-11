import math
from .tensor import Tensor
from .mtrx import mat_mul, randn

class Linear:
    def __init__(self, d_in, d_out):
        self.weights = randn((d_in, d_out))
        self.bias = 0
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # y = mx + b
        batch, seq_len = x.shape[0], x.shape[1]
        result = Tensor((batch, seq_len, self.weights.shape[-1]))
        for b in range(batch):
            result[b] = mat_mul(x[b], self.weights) #+ self.bias
        return result