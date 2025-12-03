import numpy as np


class Conv2D:
    """Simple 2D Convolutional layer using pure NumPy with manual backprop."""
    
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization
        fan_in = input_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * std
        self.bias = np.zeros(output_channels)
        
        # Cache for backward pass
        self.cache = {}
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, input_channels, height, width)
        
        Returns:
            Output of shape (batch, output_channels, out_height, out_width)
        """
        batch_size, _, height, width = x.shape
        
        # Apply padding
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        # Calculate output dimensions
        out_height = (x.shape[2] - self.kernel_size) // self.stride + 1
        out_width = (x.shape[3] - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.output_channels, out_height, out_width))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.output_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        # Extract receptive field
                        receptive_field = x[b, :, h_start:h_end, w_start:w_end]
                        
                        # Convolve with kernel and add bias
                        output[b, oc, i, j] = np.sum(receptive_field * self.weights[oc]) + self.bias[oc]
        
        # Cache for backward
        self.cache['x'] = x
        self.cache['output_shape'] = (out_height, out_width)
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output: Gradient of shape (batch, output_channels, out_height, out_width)
        
        Returns:
            grad_input: Gradient with respect to input
        """
        x = self.cache['x']
        batch_size, _, height, width = x.shape
        out_height, out_width = self.cache['output_shape']
        
        # Initialize gradients
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(x)
        
        # Compute gradients
        for b in range(batch_size):
            for oc in range(self.output_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        # Gradient for weights
                        receptive_field = x[b, :, h_start:h_end, w_start:w_end]
                        grad_weights[oc] += receptive_field * grad_output[b, oc, i, j]
                        
                        # Gradient for bias
                        grad_bias[oc] += grad_output[b, oc, i, j]
                        
                        # Gradient for input
                        grad_input[b, :, h_start:h_end, w_start:w_end] += self.weights[oc] * grad_output[b, oc, i, j]
        
        # Remove padding from grad_input if needed
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        self.grad_weights = grad_weights
        self.grad_bias = grad_bias
        
        return grad_input

class MaxPool2D:
    """Max pooling layer."""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}
    
    def forward(self, x):
        """
        Args:
            x: Input of shape (batch, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.cache['max_indices'] = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size
                        
                        window = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.max(window)
                        
                        # Store max index for backward
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        self.cache['max_indices'][b, c, i, j] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        self.cache['input_shape'] = x.shape
        return output
    
    def backward(self, grad_output):
        """Backward pass."""
        input_shape = self.cache['input_shape']
        max_indices = self.cache['max_indices']
        
        grad_input = np.zeros(input_shape)
        batch_size, channels, out_height, out_width = grad_output.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_idx, w_idx = max_indices[b, c, i, j]
                        grad_input[b, c, h_idx, w_idx] += grad_output[b, c, i, j]
        
        return grad_input

class ReLU:
    """ReLU activation."""
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, x):
        self.cache['x'] = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        x = self.cache['x']
        return grad_output * (x > 0)

class Flatten:
    """Flatten layer."""
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, x):
        self.cache['input_shape'] = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.cache['input_shape'])

class Linear:
    """Fully connected layer."""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Xavier initialization
        std = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * std
        self.bias = np.zeros(output_size)
        
        self.cache = {}
    
    def forward(self, x):
        self.cache['x'] = x
        return x @ self.weights + self.bias
    
    def backward(self, grad_output):
        x = self.cache['x']
        
        self.grad_weights = x.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        
        grad_input = grad_output @ self.weights.T
        return grad_input
