import math
import random

from .tensor import Tensor

def dot_product(v1, v2):
    weighted_sum = 0
    for i in range(len(v1.tensor)):
        weighted_sum += v1[i] * v2[i]
    return weighted_sum

def softmax(tensor):
    # for each row
    for i in range(len(tensor)):
        ek = [math.exp(v) for v in tensor[i]]
        tensor[i] = [round(e / sum(ek), 4) for e in ek]
    return tensor

def mat_mul(m1, m2):
    # check last dim of m1 equals first dim of m2, eg. m1: (3, 6) m2: (6, 3)
    assert m1.shape[-1] == m2.shape[0], f"Cannot multiply shapes {m1.shape} and {m2.shape}"
    # 1D × 1D = dot product (scalar result)
    if len(m1.shape) == 1 and len(m2.shape) == 1:
        # TODO: temporarily creating new tensors, need to modify current data
        tensor = Tensor((m2.shape[0]))
        weighted_sum = dot_product(m1, m2)
        return weighted_sum
    elif len(m1.shape) == 2 and len(m2.shape) == 2:
        tensor = zeros((m1.shape[0], m2.shape[1]))
        for i in range(m1.shape[0]):
            for j in range(m2.shape[1]):
                weighted_sum = 0
                for k in range(m1.shape[1]):
                    weighted_sum += (m1[i][k] * m2[k][j])
                tensor[i][j] = round(weighted_sum, 4)
        return tensor

def reshape(m, s):
    batch, seq_len, emb_dim = m.shape
    head_dim = s[-1]
    for b in range(batch):
        for t in range(seq_len):
            head_dims = []
            for e in range(0, emb_dim, head_dim):
                chunk = m[b][t][e:e+head_dim]
                head_dims.append(chunk)
            m[b][t] = head_dims

    m.update_shape(s)
    return m

def mask(m1):
    assert len(m1.shape) == 2, f"Cannot mask shape {m1.shape}, only 2 dim supported."
    for i in range(len(m1)):
        for j in range(i + 1, len(m1[i])):
            m1[i][j] = float('-inf')
    return m1

def randn(shape):
    return Tensor(shape, use_rand=True)

def zeros(shape):
    return Tensor(shape, v=0)

def ones(shape):
    return Tensor(shape, v=1)