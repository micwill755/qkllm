import numpy as np
import ssl
import urllib.request
from torchvision import datasets
from torchvision.transforms import ToTensor
from convolutional import Convolutional

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context

# Simple activation and pooling layers
class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, x):
        return np.reshape(x, self.output_shape)
    
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward(self, x):
        self.input = x
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, x):
        self.input = x
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

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
x_train, y_train = preprocess_data(train_dataset, 1000)
x_test, y_test = preprocess_data(test_dataset, 100)

# Build network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Activation(sigmoid, sigmoid_prime),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Activation(sigmoid, sigmoid_prime),
    Dense(100, 10),
    Activation(sigmoid, sigmoid_prime)
]

# Training
epochs = 10
learning_rate = 0.1

for epoch in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        # Forward
        output = x
        for layer in network:
            output = layer.forward(output)
        
        # Error
        error += mse(y, output)
        
        # Backward
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    
    error /= len(x_train)
    print(f"Epoch {epoch + 1}/{epochs}, Error: {error:.4f}")

# Testing
correct = 0
for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    
    if np.argmax(output) == np.argmax(y):
        correct += 1

print(f"\nTest Accuracy: {correct / len(x_test) * 100:.2f}%")
