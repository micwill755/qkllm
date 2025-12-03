import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from cnn_pytorch import SimpleCNN

def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Single GPU or CPU
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def generate_dummy_data(num_samples=100, image_size=28):
    """Generate dummy data for testing."""
    X = torch.randn(num_samples, 1, image_size, image_size)
    y = torch.randint(0, 10, (num_samples,))
    return X, y


def train(model, train_loader, criterion, optimizer, device, rank, epochs=5):
    """Training loop with FSDP."""
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
        
        # Only print from rank 0
        if rank == 0:
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def evaluate(model, test_loader, device, rank):
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


def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    if rank == 0:
        print(f"Using device: {device}")
        print(f"World size: {world_size}")
    
    # Generate dummy data
    if rank == 0:
        print("Generating dummy data...")
    X_train, y_train = generate_dummy_data(num_samples=200, image_size=28)
    X_test, y_test = generate_dummy_data(num_samples=50, image_size=28)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Use DistributedSampler for multi-GPU training
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=shuffle,
        sampler=train_sampler
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False,
        sampler=test_sampler
    )
    
    # Create model
    if rank == 0:
        print("Creating model...")
    model = SimpleCNN().to(device)
    
    # Wrap model with FSDP
    if world_size > 1:
        if rank == 0:
            print("Wrapping model with FSDP...")
        
        # Auto wrap policy: wrap layers with more than 100 parameters
        auto_wrap_policy = size_based_auto_wrap_policy
        
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # Shard parameters, gradients, and optimizer states
            device_id=device,
        )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Train
    if rank == 0:
        print("Training...")
    train(model, train_loader, criterion, optimizer, device, rank, epochs=3)
    
    # Evaluate
    if rank == 0:
        print("\nEvaluating...")
    accuracy = evaluate(model, test_loader, device, rank)
    if rank == 0:
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save model (only from rank 0)
    if rank == 0:
        if world_size > 1:
            # For FSDP, need to use special save method
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = model.state_dict()
                torch.save(state_dict, 'simple_cnn_fsdp.pth')
        else:
            torch.save(model.state_dict(), 'simple_cnn_fsdp.pth')
        
        print("\nModel saved to simple_cnn_fsdp.pth")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
