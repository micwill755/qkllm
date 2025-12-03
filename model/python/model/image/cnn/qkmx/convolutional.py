import mx
import numpy as np


class Convolutional(mx.Module):
    """
    2D Convolutional layer with full autograd support.
    
    Uses mx.conv2d() which is implemented in C with automatic differentiation.
    """
    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.kernel_size = kernel_size
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        
        # Initialize parameters with autograd support using mx functions
        # Using Xavier/He initialization for better training
        fan_in = input_depth * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)  # He initialization
        
        # Create trainable parameters using mx functions
        self.kernels = mx.Parameter(mx.randn(self.kernels_shape) * std)
        self.biases = mx.Parameter(mx.zeros((depth,)))
    
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        """
        Forward pass using mx.conv2d with autograd.
        
        Args:
            x: Input tensor of shape (input_depth, height, width)
        
        Returns:
            Output tensor of shape (depth, out_height, out_width)
        """
        # Convert input to mx.Tensor if needed
        if not isinstance(x, mx.Tensor):
            x = mx.from_numpy(x)
        
        # Build output channels - keep everything in autograd graph
        # Store intermediate results to prevent garbage collection
        self._forward_cache = []
        output_channels = []
        
        for i in range(self.depth):
            channel_sum = None
            
            # Sum convolutions over all input channels
            for j in range(self.input_depth):
                # Get the kernel for this output-input channel pair
                # IMPORTANT: Store intermediate indexing results to keep them alive
                kernel_i = self.kernels[i]
                self._forward_cache.append(kernel_i)  # Keep alive
                kernel = kernel_i[j]  # Shape: (kernel_size, kernel_size)
                self._forward_cache.append(kernel)  # Keep alive
                
                # Get the input channel
                input_channel = x[j]  # Shape: (height, width)
                self._forward_cache.append(input_channel)  # Keep alive
                
                # Perform 2D convolution (autograd tracks this!)
                conv_result = mx.conv2d(input_channel, kernel)
                
                # Accumulate - keep in autograd graph
                if channel_sum is None:
                    channel_sum = conv_result
                else:
                    channel_sum = channel_sum + conv_result
            
            # Add bias for this channel (scalar broadcasts across spatial dims)
            bias_i = self.biases[i]
            self._forward_cache.append(bias_i)  # Keep alive
            channel_with_bias = channel_sum + bias_i
            output_channels.append(channel_with_bias)
        
        # Stack channels along dimension 0 - preserves autograd!
        output = mx.stack(output_channels, dim=0)
        
        return output