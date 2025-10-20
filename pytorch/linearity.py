''' 

Non-linearity means the relationship between input and output isn't a straight line.

Linear vs Non-Linear
Linear function (straight line): f(x) = 2x + 3

- Doubling the input doubles the change in output
- Graph is always a straight line

Non-linear function (curved):
f(x) = x² or f(x) = max(0, x) (ReLU)
- Relationship is more complex
- Graph curves or bends

'''

import torch
import matplotlib.pyplot as plt

# part 1

import torch

# Simple example: 2D input -> 3D hidden -> 2D output
torch.manual_seed(42)

# Three separate linear layers (WITHOUT ReLU)
layer1 = torch.nn.Linear(2, 3, bias=False)  # W1: 3x2
layer2 = torch.nn.Linear(3, 2, bias=False)  # W2: 2x3

# Input
x = torch.tensor([[1.0, 2.0]])

# Stacked linear layers (no activation)
h = layer1(x)           # h = x @ W1.T
output_stacked = layer2(h)  # output = h @ W2.T = (x @ W1.T) @ W2.T

print("Output from stacked layers:", output_stacked)

# This is mathematically equivalent to ONE linear layer
W_combined = layer2.weight @ layer1.weight  # Combine: W2 @ W1
output_single = x @ W_combined.T

print("Output from single combined layer:", output_single)
print("Are they equal?", torch.allclose(output_stacked, output_single))

# Show the math
print("\nThe math:")
print(f"W1 shape: {layer1.weight.shape}")
print(f"W2 shape: {layer2.weight.shape}")
print(f"W_combined shape: {W_combined.shape}")
print(f"\nTwo layers: output = x @ W1.T @ W2.T")
print(f"One layer:  output = x @ (W2 @ W1).T")
print(f"These are identical!")

# part 2

x = torch.linspace(-5, 5, 100)

# Linear: y = 2x + 1
linear = 2 * x + 1
# Non-linear: ReLU
relu = torch.relu(x)
# Non-linear: x²
squared = x ** 2

# Plot all three functions
plt.figure(figsize=(12, 4))

# Linear
plt.subplot(1, 3, 1)
plt.plot(x, linear)
plt.title('Linear: y = 2x + 1')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# ReLU
plt.subplot(1, 3, 2)
plt.plot(x, relu)
plt.title('Non-linear: ReLU(x)')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# x²
plt.subplot(1, 3, 3)
plt.plot(x, squared)
plt.title('Non-linear: x²')
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

'''
When you stack linear transformations without activation, you can always combine 
the weight matrices by multiplying them together (W2 @ W1), and you'll get the exact same output with just one layer!
'''