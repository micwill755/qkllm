import numpy as np
from convolutional_numpy import Conv2D, MaxPool2D, ReLU, Flatten, Linear


class SimpleCNN:
    """Simple CNN for MNIST-like data."""
    
    def __init__(self):
        # Architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> FC -> FC
        self.conv1 = Conv2D(input_channels=1, output_channels=8, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        self.conv2 = Conv2D(input_channels=8, output_channels=16, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)
        
        self.flatten = Flatten()
        self.fc1 = Linear(16 * 7 * 7, 128)  # For 28x28 input -> 7x7 after 2 poolings
        self.relu3 = ReLU()
        self.fc2 = Linear(128, 10)
        
        self.layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten, self.fc1, self.relu3, self.fc2
        ]
    
    def forward(self, x):
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        """Backward pass through all layers."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def get_parameters(self):
        """Get all trainable parameters."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.append((layer, 'weights', layer.weights))
            if hasattr(layer, 'bias'):
                params.append((layer, 'bias', layer.bias))
        return params


def softmax(x):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(predictions, targets):
    """
    Cross entropy loss.
    
    Args:
        predictions: Raw logits of shape (batch, num_classes)
        targets: One-hot encoded targets of shape (batch, num_classes)
    
    Returns:
        loss: Scalar loss
        grad: Gradient with respect to predictions
    """
    batch_size = predictions.shape[0]
    
    # Softmax
    probs = softmax(predictions)
    
    # Loss
    loss = -np.sum(targets * np.log(probs + 1e-8)) / batch_size
    
    # Gradient
    grad = (probs - targets) / batch_size
    
    return loss, grad


def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoding."""
    batch_size = len(labels)
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), labels] = 1
    return one_hot


class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def step(self, model):
        """Update parameters."""
        for layer, param_name, param in model.get_parameters():
            grad = getattr(layer, f'grad_{param_name}')
            param -= self.learning_rate * grad


def generate_dummy_data(num_samples=100, image_size=28):
    """Generate dummy data for testing."""
    X = np.random.randn(num_samples, 1, image_size, image_size).astype(np.float32)
    y = np.random.randint(0, 10, num_samples)
    return X, y


def train(model, X_train, y_train, epochs=5, batch_size=32, learning_rate=0.01):
    """Training loop."""
    optimizer = SGD(learning_rate=learning_rate)
    num_samples = len(X_train)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(num_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Forward pass
            predictions = model.forward(batch_X)
            
            # Compute loss
            targets = one_hot_encode(batch_y)
            loss, grad_loss = cross_entropy_loss(predictions, targets)
            
            # Backward pass
            model.backward(grad_loss)
            
            # Update parameters
            optimizer.step(model)
            
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def evaluate(model, X_test, y_test, batch_size=32):
    """Evaluate model accuracy."""
    num_samples = len(X_test)
    correct = 0
    
    for i in range(0, num_samples, batch_size):
        batch_X = X_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        
        predictions = model.forward(batch_X)
        predicted_classes = np.argmax(predictions, axis=1)
        
        correct += np.sum(predicted_classes == batch_y)
    
    accuracy = correct / num_samples
    return accuracy


if __name__ == "__main__":
    print("Creating simple CNN...")
    model = SimpleCNN()
    
    print("Generating dummy data...")
    X_train, y_train = generate_dummy_data(num_samples=200, image_size=28)
    X_test, y_test = generate_dummy_data(num_samples=50, image_size=28)
    
    print("Training...")
    train(model, X_train, y_train, epochs=3, batch_size=16, learning_rate=0.01)
    
    print("\nEvaluating...")
    accuracy = evaluate(model, X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
