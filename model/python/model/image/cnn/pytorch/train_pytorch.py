import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from cnn_pytorch import SimpleCNN

def generate_dummy_data(num_samples=100, image_size=28):
    """Generate dummy data for testing."""
    X = torch.randn(num_samples, 1, image_size, image_size)
    y = torch.randint(0, 10, (num_samples,))
    return X, y


def train(model, train_loader, criterion, optimizer, device, epochs=5):
    """Training loop."""
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def evaluate(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate dummy data
    print("Generating dummy data...")
    X_train, y_train = generate_dummy_data(num_samples=200, image_size=28)
    X_test, y_test = generate_dummy_data(num_samples=50, image_size=28)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = SimpleCNN().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Train
    print("Training...")
    train(model, train_loader, criterion, optimizer, device, epochs=3)
    
    # Evaluate
    print("\nEvaluating...")
    accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'simple_cnn.pth')
    print("\nModel saved to simple_cnn.pth")
