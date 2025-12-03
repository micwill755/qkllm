import sys
sys.path.insert(0, '/Users/mcwlm/Documents/code/deep learning/llm/qkmx/src')

# Force reload if mx was already imported
if 'mx' in sys.modules:
    import importlib
    importlib.reload(sys.modules['mx'])

import numpy as np
import ssl
import mx
from torchvision import datasets
from torchvision.transforms import ToTensor
from convolutional import Convolutional

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

# Simple layers using mx with autograd
class Reshape(mx.Module):
    """Reshape layer - just changes tensor shape"""
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return x.reshape(self.output_shape)

class Dense(mx.Module):
    """Fully connected layer with autograd"""
    def __init__(self, input_size, output_size):
        super().__init__()
        # Initialize with small random values using mx functions
        std = np.sqrt(2.0 / input_size)
        self.weights = mx.Parameter(mx.randn((output_size, input_size)) * std)
        self.bias = mx.Parameter(mx.zeros((output_size, 1)))
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        return self.weights.matmul(x) + self.bias


class Activation(mx.Module):
    """GELU activation with full autograd support"""
    def __call__(self, x):
        return mx.gelu(x)


def mse_loss(y_pred, y_true):
    """Mean squared error loss with autograd support"""
    diff = y_pred - y_true
    squared = diff * diff
    # mean() now returns a tensor with autograd support!
    return squared.mean()

# Preprocess data
def preprocess_data(dataset, limit):
    x_data = []
    y_data = []
    for i, (img, label) in enumerate(dataset):
        if i >= limit:
            break
        x_data.append(img.numpy())
        y_label = np.zeros((10, 1))
        y_label[label] = 1
        y_data.append(y_label)
    return np.array(x_data), np.array(y_data)

def generate_synthetic_data(n_samples):
    """Generate synthetic 28x28 images for testing"""
    x_data = []
    y_data = []
    for i in range(n_samples):
        # Create simple patterns for each digit
        label = i % 10
        img = np.random.rand(1, 28, 28) * 0.3
        # Add some pattern based on label
        img[0, label*2:label*2+5, label*2:label*2+5] = 0.8
        x_data.append(img)
        y_label = np.zeros((10, 1))
        y_label[label] = 1
        y_data.append(y_label)
    return np.array(x_data), np.array(y_data)

# Load and prepare data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
# Use fewer samples for faster testing
x_train, y_train = preprocess_data(train_dataset, 100)
x_test, y_test = preprocess_data(test_dataset, 20)

# Build network - SIMPLIFIED WITHOUT CONV for debugging
print("\n=== Building simplified network (no conv) ===")
network = [
    Reshape((1, 28, 28), (28 * 28, 1)),
    Dense(28 * 28, 100),
    Activation(),  # GELU activation with autograd
    Dense(100, 10)
    # No activation at the end - softmax is implicit in loss for numerical stability
]

# # Original network with conv (currently has issues)
# network = [
#     Convolutional((1, 28, 28), 3, 5),
#     Activation(),  # GELU activation with autograd
#     Reshape((5, 26, 26), (5 * 26 * 26, 1)),
#     Dense(5 * 26 * 26, 100),
#     Activation(),  # GELU activation with autograd
#     Dense(100, 10)
#     # No activation at the end - softmax is implicit in loss for numerical stability
# ]

# Collect all parameters for optimizer
all_params = []
for layer in network:
    if hasattr(layer, 'parameters'):
        all_params.extend(layer.parameters())

# Create optimizer with much smaller learning rate
optimizer = mx.SGD(all_params, learning_rate=0.001)

# Training with autograd!
epochs = 3  # Fewer epochs for testing

print("\n=== Starting Training ===")
print(f"Training samples: {len(x_train)}")
print(f"Network layers: {len(network)}")

for epoch in range(epochs):
    total_loss = 0.0  # Use float to accumulate
    
    for i, (x, y) in enumerate(zip(x_train, y_train)):
        if i % 10 == 0:
            print(f"Epoch {epoch + 1}, Sample {i}/{len(x_train)}", end='\r')
        
        # Convert to mx tensors using mx.array()
        x_tensor = mx.array(x)
        y_tensor = mx.array(y)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = x_tensor
        for layer in network:
            output = layer(output)
        
        # Compute loss
        loss = mse_loss(output, y_tensor)
        
        # Extract scalar value for logging
        loss_val = loss.item()
        total_loss += loss_val
        
        # DEBUG: Check first sample
        if epoch == 0 and i == 0:
            print(f"\n=== DEBUG: First sample ===")
            print(f"Loss: {loss_val:.4f}")
            print(f"Loss requires_grad: {loss.requires_grad}")
            print(f"Output requires_grad: {output.requires_grad}")
            print("Starting backward pass...")
        
        # Backward pass (autograd!)
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Backward pass timed out")
            
            # Set 2 second timeout for backward pass
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(2)
            
            loss.backward()
            
            signal.alarm(0)  # Cancel alarm
            
        except TimeoutError:
            print(f"\n!!! Backward pass timed out on sample {i} !!!")
            print("This indicates an issue with the autograd implementation.")
            print("Skipping backward pass for this sample.")
            continue
        
        # DEBUG: Check gradients on first sample
        if epoch == 0 and i == 0:
            dense_layer = network[1]  # First Dense layer
            print(f"Backward pass completed!")
            print(f"Dense weights grad: {dense_layer.weights.grad}")
            print(f"Dense bias grad: {dense_layer.bias.grad}")
            if dense_layer.weights.grad:
                print(f"Dense weights grad mean: {dense_layer.weights.grad.mean().item():.6f}")
        
        # Update weights
        optimizer.step()
        
        # DEBUG: Check if weights changed
        if epoch == 0 and i == 0:
            print(f"Dense weights mean after step: {dense_layer.weights.mean().item():.6f}")
    
    avg_loss = total_loss / len(x_train)
    print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# Testing
correct = 0
for x, y in zip(x_test, y_test):
    x_tensor = mx.array(x)
    
    # Forward pass
    output = x_tensor
    for layer in network:
        output = layer(output)
    
    # Get prediction - convert to numpy for argmax
    output_np = output.numpy()
    if np.argmax(output_np) == np.argmax(y):
        correct += 1

print(f"\nTest Accuracy: {correct / len(x_test) * 100:.2f}%")
