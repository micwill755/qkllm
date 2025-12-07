import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from cnn_pytorch import SimpleCNN


def setup_ddp():
    """Initialize DDP environment."""
    # Get rank and world size from environment variables set by torchrun
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        # Initialize process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
        )
    
    return rank, local_rank, world_size


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def generate_dummy_data(num_samples=100, image_size=28):
    """Generate dummy data for testing."""
    X = torch.randn(num_samples, 1, image_size, image_size)
    y = torch.randint(0, 10, (num_samples,))
    return X, y


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank):
    """Train one epoch."""
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
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
    
    # Average metrics across all processes
    if dist.is_initialized():
        # Convert to tensors for all_reduce
        metrics = torch.tensor([epoch_loss, correct, total], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        epoch_loss, correct, total = metrics.cpu().numpy()
        epoch_loss /= world_size  # Average loss across processes
    
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


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
    
    # Aggregate metrics across all processes
    if dist.is_initialized():
        metrics = torch.tensor([correct, total], dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        correct, total = metrics.cpu().numpy()
    
    accuracy = 100 * correct / total
    return accuracy


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir='checkpoints', rank=0):
    """Save checkpoint (only rank 0)."""
    if rank != 0:
        return None
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # For DDP, save the underlying model (not the DDP wrapper)
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Keep only last 3 checkpoints
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')])
    if len(checkpoints) > 3:
        for old_ckpt in checkpoints[:-3]:
            os.remove(os.path.join(checkpoint_dir, old_ckpt))
    
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_dir='checkpoints'):
    """Load latest checkpoint if exists."""
    if not os.path.exists(checkpoint_dir):
        return 0, None
    
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')])
    if not checkpoints:
        return 0, None
    
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    # Load model state (handle DDP wrapper)
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def main():
    # Setup DDP
    rank, local_rank, world_size = setup_ddp()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    if rank == 0:
        print(f"DDP Training")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
    
    # Generate dummy data
    if rank == 0:
        print("Generating dummy data...")
    X_train, y_train = generate_dummy_data(num_samples=200, image_size=28)
    X_test, y_test = generate_dummy_data(num_samples=50, image_size=28)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create samplers for distributed training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        test_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=2
    )
    
    # Create model
    if rank == 0:
        print("Creating model...")
    model = SimpleCNN().to(device)
    
    # Wrap model with DDP
    if world_size > 1:
        if rank == 0:
            print("Wrapping model with DDP...")
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # Set to True if you have unused parameters
        )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Checkpoint directory
    checkpoint_dir = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
    
    # Try to load checkpoint
    start_epoch = 0
    if rank == 0:
        print(f"Checkpoint directory: {checkpoint_dir}")
        print("Checking for existing checkpoints...")
    
    if os.path.exists(checkpoint_dir):
        start_epoch, last_loss = load_checkpoint(model, optimizer, checkpoint_dir)
        if start_epoch > 0 and rank == 0:
            print(f"Resuming from epoch {start_epoch}, last loss: {last_loss:.4f}")
    
    # Synchronize start_epoch across all processes
    if world_size > 1:
        start_epoch_tensor = torch.tensor(start_epoch, device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
    
    # Training loop
    epochs = 5
    if rank == 0:
        print("Training...")
    
    for epoch in range(start_epoch, epochs):
        # Set epoch for sampler (important for proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train one epoch
        avg_loss, accuracy = train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Save checkpoint
            checkpoint_path = save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_dir, rank)
            if checkpoint_path:
                print(f"Checkpoint saved: {checkpoint_path}")
    
    # Evaluate
    if rank == 0:
        print("\nEvaluating...")
    accuracy = evaluate(model, test_loader, device, rank)
    if rank == 0:
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save final model
    if rank == 0:
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save(model_state, 'simple_cnn_ddp_final.pth')
        print("\nFinal model saved to simple_cnn_ddp_final.pth")
    
    # Cleanup
    cleanup_ddp()


if __name__ == "__main__":
    main()