import math
import random
from itertools import product

from .tensor import Tensor

def dot_product(v1, v2):
    weighted_sum = 0
    for i in range(len(v1.tensor)):
        weighted_sum += v1[i] * v2[i]
    return weighted_sum

def mat_mul(m1, m2):
    assert m1.shape[-1] == m2.shape[-2], f"Cannot multiply shapes {m1.shape} and {m2.shape}"
    
    if len(m1.shape) == 1 and len(m2.shape) == 1:
        return dot_product(m1, m2)
    
    # Create output shape: replace last 2 dims with m1[-2] and m2[-1]
    out_shape = list(m1.shape[:-1]) + [m2.shape[-1]]
    tensor = zeros(tuple(out_shape))
    
    # Generate all batch indices (everything except last 2 dims)
    batch_dims = m1.shape[:-2]
    batch_indices = product(*[range(d) for d in batch_dims]) if batch_dims else [()]
    
    # For each batch, do 2D matmul on last 2 dims
    for batch_idx in batch_indices:
        for i in range(m1.shape[-2]):
            for j in range(m2.shape[-1]):
                weighted_sum = 0
                for k in range(m1.shape[-1]):
                    idx1 = batch_idx + (i, k)
                    idx2 = batch_idx + (k, j)
                    weighted_sum += m1[idx1] * m2[idx2]
                tensor[batch_idx + (i, j)] = round(weighted_sum, 4)
    
    return tensor

def reshape(m, s):
    batch_dim = m.shape[:2]
    batch_indices = product(*[range(d) for d in batch_dim]) if batch_dim else [()]

    if len(m.shape) == 3:
        emb_dim = m.shape[-1]
        head_dim = s[-1]
        for batch_idx in batch_indices:
            b, t = batch_idx
            head_dims = []
            for e in range(0, emb_dim, head_dim):
                chunk = m[b][t][e:e+head_dim]
                head_dims.append(chunk)
            m[b][t] = head_dims
        m.update_shape(s)
    elif len(m.shape) == 4:
        num_heads, head_dim = m.shape[2:]
        for batch_idx in batch_indices:
            b, t = batch_idx
            embedding = []
            for h in range(num_heads):
                for e in range(len(m[b][t][h])):
                    embedding.append(m[b][t][h][e])
            m[b][t] = embedding
        m.update_shape(s)
    return m

def softmax(tensor):
    # Apply softmax on last dimension for all batch dims
    batch_dims = tensor.shape[:-1]
    batch_indices = product(*[range(d) for d in batch_dims])
    
    for batch_idx in batch_indices:
        row = tensor[batch_idx]
        ek = [math.exp(v) for v in row]
        tensor[batch_idx] = [round(e / sum(ek), 4) for e in ek]
    return tensor

def mask(m1):
    # Mask last 2 dimensions (upper triangle) for all batch dims
    batch_dims = m1.shape[:-2]
    batch_indices = product(*[range(d) for d in batch_dims]) if batch_dims else [()]
    seq_len = m1.shape[-1]
    
    for batch_idx in batch_indices:
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                m1[batch_idx + (i, j)] = float('-inf')
    return m1

def randn(shape):
    return Tensor(shape, use_rand=True)

def zeros(shape):
    return Tensor(shape, v=0)

def ones(shape):
    return Tensor(shape, v=1)