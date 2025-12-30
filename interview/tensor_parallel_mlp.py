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
        self.tp_size = tp_size
        
        # Split fc1 weights and bias column-wise (output dimension)
        hidden_size = fc1_weights.shape[1]
        chunk_size = hidden_size // tp_size
        
        self.fc1_shards = []
        for i in range(tp_size):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            shard_weights = fc1_weights[:, start_idx:end_idx]
            shard_bias = fc1_bias[:, start_idx:end_idx]
            self.fc1_shards.append(Linear(shard_weights, shard_bias))
        
        # Split fc2 weights row-wise (input dimension)
        self.fc2_shards = []
        for i in range(tp_size):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            shard_weights = fc2_weights[start_idx:end_idx, :]
            # Only first shard gets bias to avoid duplication
            shard_bias = fc2_bias if i == 0 else np.zeros_like(fc2_bias)
            self.fc2_shards.append(Linear(shard_weights, shard_bias))
        
        self.relu_shards = [ReLU() for _ in range(tp_size)]

    def forward(self, x):
        # FC1: Column parallel - each shard produces part of hidden dimension
        h_shards = []
        for i in range(self.tp_size):
            h_shard = self.fc1_shards[i].forward(x)
            h_shards.append(h_shard)
        
        # ReLU: Apply to each shard independently
        h_relu_shards = []
        for i in range(self.tp_size):
            h_relu_shard = self.relu_shards[i].forward(h_shards[i])
            h_relu_shards.append(h_relu_shard)
        
        # FC2: Row parallel - each shard processes its part, then sum results
        y_shards = []
        for i in range(self.tp_size):
            y_shard = self.fc2_shards[i].forward(h_relu_shards[i])
            y_shards.append(y_shard)
        
        # All-reduce: Sum outputs from all shards
        y = sum(y_shards)
        return y

    def backward(self, grad_output):
        # FC2 backward: Each shard gets the same grad_output
        grad_h_relu_shards = []
        for i in range(self.tp_size):
            grad_h_relu_shard = self.fc2_shards[i].backward(grad_output)
            grad_h_relu_shards.append(grad_h_relu_shard)
        
        # ReLU backward: Apply to each shard independently
        grad_h_shards = []
        for i in range(self.tp_size):
            grad_h_shard = self.relu_shards[i].backward(grad_h_relu_shards[i])
            grad_h_shards.append(grad_h_shard)
        
        # FC1 backward: Each shard computes grad_input, then sum (all-reduce)
        grad_x_shards = []
        for i in range(self.tp_size):
            grad_x_shard = self.fc1_shards[i].backward(grad_h_shards[i])
            grad_x_shards.append(grad_x_shard)
        
        # All-reduce: Sum gradients from all shards
        grad_x = sum(grad_x_shards)
        return grad_x

    def update(self, learning_rate):
        for i in range(self.tp_size):
            self.fc1_shards[i].update(learning_rate)
            self.fc2_shards[i].update(learning_rate)


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
