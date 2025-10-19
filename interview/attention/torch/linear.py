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
        y = mat_mul(x, self.weights) #+ self.bias
        return y