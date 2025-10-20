import math
import torch.torch
from torch.tensor import Tensor
from torch.torch import softmax, dot_product, mat_mul, mask, reshape
from torch.linear import Linear

#inputs = torch.rand(6, 3)
# we are hard coding to see the values 
inputs = Tensor([[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
    )

# 1. compute context vector for a single query token which is journey

'''att_scores = Tensor((inputs.shape[0], inputs.shape[0]))
for i in range(len(inputs)):
    for j in range(len(inputs)):
        att_scores[i][j] = round(dot_product(inputs[i], inputs[j]), 4)

print(att_scores)
att_weights = softmax(att_scores)

query = inputs[1]
context = torch.torch.zeros(query.shape)

for i in range(context.shape[0]):
    weighted_sum = 0
    for j in range(len(att_weights)):
        weighted_sum += att_weights[1][j] * inputs[j][i]
    context[i] = round(weighted_sum, 4)

print(context)'''

# 2. compute attention weights for all tokens using matrix multiplication
'''context = torch.torch.zeros(inputs.shape)
att_scores = mat_mul(inputs, inputs.T)
print(inputs.T)
att_weights = softmax(att_scores)
context = mat_mul(att_weights, inputs)
print(context)'''

# implement self attention with trainable weights
class SelfAttention:
    def __init__(self, d_in, d_out):
        self.q_W = Linear(d_in, d_out)
        self.k_W = Linear(d_in, d_out)
        self.v_W = Linear(d_in, d_out)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        att_scores = mat_mul(query, key.T)
        att_weights = softmax(att_scores)
        context = mat_mul(att_weights, value)
        return context
    
'''attention = SelfAttention(inputs.shape[1], inputs.shape[1])
out = attention(inputs)
print(out)'''

class SingleHeadCasualAttention:
    def __init__(self, d_in, d_out):
        self.q_W = Linear(d_in, d_out)
        self.k_W = Linear(d_in, d_out)
        self.v_W = Linear(d_in, d_out)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        att_scores = mat_mul(query, key.T)
        att_scores = mask(att_scores)
        att_weights = softmax(att_scores)
        context = mat_mul(att_weights, value)
        return context

'''attention = SingleHeadCasualAttention(inputs.shape[1], inputs.shape[1])
out = attention(inputs)
print(out)'''

class MultiHeadAttention:
    def __init__(self, emd_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = emd_dim // num_heads

        self.q_W = Linear(emd_dim, emd_dim)
        self.k_W = Linear(emd_dim, emd_dim)
        self.v_W = Linear(emd_dim, emd_dim)

        self.out_proj = Linear(emd_dim, emd_dim)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # [batch, seq_len, emb_dim]
        batch, seq_len, emb_dim = x.shape

        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        # reshape 
        queries = reshape(query, (batch, seq_len, self.num_heads, self.head_dim))
        keys = reshape(key, (batch, seq_len, self.num_heads, self.head_dim))
        value = reshape(value, (batch, seq_len, self.num_heads, self.head_dim))

        att_scores = mat_mul(query, key.T) // (self.emb_dim ** 0.5)

        att_scores = mask(att_scores)
        att_weights = softmax(att_scores)
        context = mat_mul(att_weights, value)
        # combines heads

        return self.out_proj(context)

emb_dim = 24
inputs = torch.torch.randn((1, 8, emb_dim))
attention = MultiHeadAttention(emb_dim, num_heads=4)
out = attention(inputs)
print(out)