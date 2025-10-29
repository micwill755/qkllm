# Implement Tensor Parallel version of the MLPBlock

import numpy as np

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.x > 0)

class Linear:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

        self.d_weights = np.zeros_like(self.weights)
        self.d_bias = np.zeros_like(self.bias)
        self.x = None

    def forward(self, x):
        self.x = x
        return x.dot(self.weights) + self.bias

    def backward(self, grad_output):
        self.d_weights = self.x.T.dot(grad_output)
        self.d_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output.dot(self.weights.T)
        return grad_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias

class MLPBlock:
    def __init__(self, fc1_weights, fc1_bias, fc2_weights, fc2_bias):
        self.fc1 = Linear(fc1_weights, fc1_bias)
        self.relu = ReLU()
        self.fc2 = Linear(fc2_weights, fc2_bias)

    def forward(self, x):
        h = self.fc1.forward(x)
        h_relu = self.relu.forward(h)
        y = self.fc2.forward(h_relu)
        return y

    def backward(self, grad_output):
        grad_h_relu = self.fc2.backward(grad_output)
        grad_h = self.relu.backward(grad_h_relu)
        grad_x = self.fc1.backward(grad_h)
        return grad_x

    def update(self, learning_rate):
        self.fc1.update(learning_rate)
        self.fc2.update(learning_rate)




class TensorParallelMLPBlock:
    def __init__(self, fc1_weights, fc1_bias, fc2_weights, fc2_bias, tp_size=2):
        fc1_weights_shard_size = len(fc1_weights) / 2
        print('Michae;', fc1_weights_shard_size)
        shard0 = fc1_weights[fc1_weights_shard_size:]
        shard1 = fc1_weights[:fc1_weights_shard_size]

        pass

    def forward(self, x):
        pass

    def backward(self, grad_output):
        pass

    def update(self, learning_rate):
        pass

if __name__ == "__main__":  
    # --- 1. Set Parameters ---
    B, C_in, C_hidden, C_out = 4, 8, 32, 8
    learning_rate = 0.01
    TP_SIZE = 2
    np.random.seed(42)

    fc1_weights = np.random.randn(C_in, C_hidden) * 0.01
    fc1_bias = np.random.randn(1, C_hidden) * 0.01
    fc2_weights = np.random.randn(C_hidden, C_out) * 0.01
    fc2_bias = np.random.randn(1, C_out) * 0.01

    seq_model = MLPBlock(
        fc1_weights.copy(), fc1_bias.copy(), 
        fc2_weights.copy(), fc2_bias.copy()
    )
    
    tp_model = TensorParallelMLPBlock(
        fc1_weights.copy(), fc1_bias.copy(), 
        fc2_weights.copy(), fc2_bias.copy(), 
        tp_size=TP_SIZE
    )

    np.random.seed(123) # Use a different seed for data
    x = np.random.randn(B, C_in)
    grad_output = np.random.randn(B, C_out)

    print("\nTesting Forward Pass...")
    seq_output = seq_model.forward(x)
    tp_output = tp_model.forward(x)
    
    assert np.allclose(seq_output, tp_output, atol=1e-9)

    print("\nTesting Backward Pass...")
    
    seq_grad_input = seq_model.backward(grad_output)
    tp_grad_input = tp_model.backward(grad_output)
    
    assert np.allclose(seq_grad_input, tp_grad_input, atol=1e-9)

    print("Test Passed")

'''
fc1 is column-parallel: split along its output/hidden dimension; each shard keeps its own bias.

fc2 is row-parallel: split along its input/hidden rows; shard outputs are summed; fc2_bias is held once and added after the sum.

Backward:

For fc2: dW2_i = h_iᵀ @ dy, db2 = sum(dy), dh_i = dy @ W2_iᵀ.

ReLU per shard.

For fc1: dW1_i = xᵀ @ dh_pre_i, db1_i = sum(dh_pre_i), and dx is the sum over shards of dh_pre_i @ W1_iᵀ.

'''
class TensorParallelMLPBlock:
    """
    Column-parallel fc1 (split hidden/out features) +
    Row-parallel fc2 (split hidden/in rows); bias2 is applied once after the reduce-sum.

    Shapes:
      x:          (B, C_in)
      fc1 W:      (C_in, C_hidden)  -> split along axis=1 into shards
      fc1 b:      (1, C_hidden)     -> split along axis=1 into shards
      ReLU:       per-shard
      fc2 W:      (C_hidden, C_out) -> split along axis=0 into matching shards
      fc2 b:      (1, C_out)        -> kept whole, added once after sum(y_i)
    """
    def __init__(self, fc1_weights, fc1_bias, fc2_weights, fc2_bias, tp_size=2):
        assert fc1_weights.ndim == 2 and fc2_weights.ndim == 2
        assert fc1_bias.shape[1] == fc1_weights.shape[1]
        assert fc2_weights.shape[0] == fc1_weights.shape[1]
        self.tp_size = tp_size

        C_in, C_hidden = fc1_weights.shape
        # Split hidden dim indices as evenly as possible
        hidden_splits = np.array_split(np.arange(C_hidden), tp_size)

        # Sharded params
        self.W1 = [fc1_weights[:, idx] for idx in hidden_splits]          # (C_in, h_i)
        self.b1 = [fc1_bias[:, idx] for idx in hidden_splits]             # (1, h_i)
        self.W2 = [fc2_weights[idx, :] for idx in hidden_splits]          # (h_i, C_out)

        # Single shared bias for row-parallel fc2
        self.b2 = fc2_bias.copy()                                         # (1, C_out)

        # Grads (same structure)
        self.dW1 = [np.zeros_like(w) for w in self.W1]
        self.db1 = [np.zeros_like(b) for b in self.b1]
        self.dW2 = [np.zeros_like(w) for w in self.W2]
        self.db2 = np.zeros_like(self.b2)

        # Caches
        self.x = None
        self.h_pre = [None]*tp_size   # pre-activation for ReLU (per shard)
        self.h = [None]*tp_size       # post-activation (per shard)

    def forward(self, x):
        self.x = x
        # fc1 per shard -> ReLU per shard
        for i in range(self.tp_size):
            h_pre_i = x.dot(self.W1[i]) + self.b1[i]          # (B, h_i)
            self.h_pre[i] = h_pre_i
            self.h[i] = np.maximum(0.0, h_pre_i)              # ReLU

        # fc2 row-parallel: y_i = h_i @ W2_i ; then sum_i y_i, then add b2 once
        y = np.zeros((x.shape[0], self.b2.shape[1]), dtype=x.dtype)
        for i in range(self.tp_size):
            y += self.h[i].dot(self.W2[i])                    # (B, C_out)
        y += self.b2
        return y

    def backward(self, grad_output):
        """
        grad_output: (B, C_out)
        Returns grad_input dx: (B, C_in)
        """
        B = grad_output.shape[0]

        # fc2 row-parallel grads:
        # dW2_i = h_i^T @ dy ; db2 = sum(dy) once ; dh_i = dy @ W2_i^T
        for i in range(self.tp_size):
            self.dW2[i] = self.h[i].T.dot(grad_output)        # (h_i, C_out)
        self.db2 = np.sum(grad_output, axis=0, keepdims=True) # (1, C_out)

        # Backprop to h_i through ReLU
        dx = np.zeros((B, self.x.shape[1]), dtype=self.x.dtype)
        for i in range(self.tp_size):
            dh_i = grad_output.dot(self.W2[i].T)              # (B, h_i)
            # ReLU backward
            dh_pre_i = dh_i * (self.h_pre[i] > 0)             # (B, h_i)

            # fc1 column-parallel grads and dx contribution
            self.dW1[i] = self.x.T.dot(dh_pre_i)              # (C_in, h_i)
            self.db1[i] = np.sum(dh_pre_i, axis=0, keepdims=True)  # (1, h_i)
            dx += dh_pre_i.dot(self.W1[i].T)                  # accumulate (B, C_in)

        return dx

    def update(self, learning_rate):
        # SGD step on all shards + shared bias2
        for i in range(self.tp_size):
            self.W1[i] -= learning_rate * self.dW1[i]
            self.b1[i] -= learning_rate * self.db1[i]
            self.W2[i] -= learning_rate * self.dW2[i]
        self.b2 -= learning_rate * self.db2
