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
        C_in, C_hidden = fc1_weights.shape 

         # Split hidden dimension evenly across shards
        hidden_per_shard = C_hidden // tp_size

        # Column-parallel fc1: split along hidden dimension (axis=1)
        self.W1 = []
        self.b1 = []
        for i in range(tp_size):
            start_idx = i * hidden_per_shard
            end_idx = (i + 1) * hidden_per_shard
            self.W1.append(fc1_weights[:, start_idx:end_idx]) 
            self.b1.append(fc1_bias[:, start_idx:end_idx]) 

        # Row-parallel fc2: split along input dimension (axis=0) to match fc1 sharding
        self.W2 = []
        for i in range(tp_size):
            start_idx = i * hidden_per_shard
            end_idx = (i + 1) * hidden_per_shard
            self.W2.append(fc2_weights[start_idx:end_idx, :])  # (hidden_per_shard, C_out)

        # fc2 bias is shared (added once after sum)
        self.b2 = fc2_bias.copy()
        
        # Cache for backward pass
        self.x = None
        self.h_pre = [None] * tp_size  # pre-ReLU activations
        self.h = [None] * tp_size      # post-ReLU activations

    def forward(self, x):
        # TODO
        pass

    def backward(self, grad_output):
        pass

    def update(self, learning_rate):
        pass

class PipelineParallelMLPBlock:
    def __init__ (self, fc1_weights, fc1_bias, fc2_weights, fc2_bias, pp_size=2):
        self.pp_size = pp_size
        self.layers = [
            Linear(fc1_weights, fc1_bias), 
            ReLU(), 
            Linear(fc2_weights, fc2_bias)]
        
        num_layers = len(self.layers)
        layers_per_stage = num_layers // pp_size
        # depending on pipeline size, layers dont divide equally
        # so we add an additional layer to first stages
        remainder = num_layers % pp_size

        self.stages = []
        layer_idx = 0

        # distribute layers across stages
        for i in range(pp_size):
            # first 'remainder' stages get an extra layer
            stage_size = layers_per_stage + (1 if i < remainder else 0)
            # Slice layers for this stage
            stage_layers = self.layers[layer_idx:layer_idx + stage_size]
            self.stages.append(stage_layers)
            
            layer_idx += stage_size
                    
    def forward(self, x):
        # TODO
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
    PP_SIZE = 2
    np.random.seed(42)

    fc1_weights = np.random.randn(C_in, C_hidden) * 0.01
    fc1_bias = np.random.randn(1, C_hidden) * 0.01
    fc2_weights = np.random.randn(C_hidden, C_out) * 0.01
    fc2_bias = np.random.randn(1, C_out) * 0.01

    seq_model = MLPBlock(
        fc1_weights.copy(), fc1_bias.copy(), 
        fc2_weights.copy(), fc2_bias.copy()
    )
    
    # part 1 - first we are going to create seperate MLP objects to handle parallelism seperately
    # part 2 - then we will create one MLP block that can handle parallelism strategies cohesively
    tp_model = TensorParallelMLPBlock(
        fc1_weights.copy(), fc1_bias.copy(), 
        fc2_weights.copy(), fc2_bias.copy(), 
        tp_size=TP_SIZE
    )

    fc_weights = [fc1_weights.copy(), fc2_weights.copy()]
    fc_bias = [fc1_bias.copy(), fc2_bias.copy()]

    pp_model = PipelineParallelMLPBlock(
        fc1_weights.copy(), fc1_bias.copy(), 
        fc2_weights.copy(), fc2_bias.copy(), 
        pp_size=PP_SIZE
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