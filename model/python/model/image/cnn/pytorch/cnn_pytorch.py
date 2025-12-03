import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST-like data using PyTorch."""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)           # (1, 28, 28) -> (8, 28, 28)
        x = F.relu(x)               # ReLU activation
        x = self.pool1(x)           # (8, 28, 28) -> (8, 14, 14)
        
        # Conv block 2
        x = self.conv2(x)           # (8, 14, 14) -> (16, 14, 14)
        x = F.relu(x)               # ReLU activation
        x = self.pool2(x)           # (16, 14, 14) -> (16, 7, 7)
        
        # Flatten
        x = x.view(x.size(0), -1)   # (16, 7, 7) -> (784,)
        
        # Fully connected layers
        x = self.fc1(x)             # (784,) -> (128,)
        x = F.relu(x)               # ReLU activation
        x = self.fc2(x)             # (128,) -> (10,)
        
        return x
