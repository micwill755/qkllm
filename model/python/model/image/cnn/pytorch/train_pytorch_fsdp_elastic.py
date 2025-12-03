import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from cnn_pytorch import SimpleCNN
import time


def setup_distributed():
    """Initialize distributed training with elastic support."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        # Use c10d rendezvous for elastic training
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',  # Uses MASTER_ADDR and MASTER_PORT
        )
    
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


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir='checkpoints', rank=0):
    """Save checkpoint for fault tolerance.
    
    Note: checkpoint_dir should be on a shared filesystem (NFS, Lustre, etc.)
    accessible by all nodes. Only rank 0 saves to avoid conflicts.
    """
    if rank != 0:
        return None
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get full state dict from FSDP
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()
    
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
    
    # Load model state
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank):
    """Train one epoch with error handling."""
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    try:
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Periodic health check
            if batch_idx % 10 == 0 and dist.is_initialized():
                # Check if all ranks are still alive
                try:
                    dist.barrier(timeout=10)  # 10 second timeout
                except Exception as e:
                    if rank == 0:
                        print(f"Detected node failure at batch {batch_idx}: {e}")
                    raise
        
        return epoch_loss / len(train_loader), 100 * correct / total
    
    except Exception as e:
        if rank == 0:
            print(f"Error during training epoch {epoch}: {e}")
        raise


def train(model, train_loader, criterion, optimizer, device, rank, start_epoch=0, epochs=5, checkpoint_dir='checkpoints'):
    """Training loop with checkpointing and fault tolerance.
    
    Args:
        checkpoint_dir: Path to shared filesystem directory for checkpoints.
                       Should be accessible by all nodes (e.g., /shared/checkpoints)
    """
    
    for epoch in range(start_epoch, epochs):
        try:
            avg_loss, accuracy = train_epoch(model, train_loader, criterion, optimizer, device, epoch, rank)
            
            if rank == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
                
                # Save checkpoint after each epoch (only rank 0)
                checkpoint_path = save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_dir, rank)
                if checkpoint_path:
                    print(f"Checkpoint saved: {checkpoint_path}")
        
        except Exception as e:
            if rank == 0:
                print(f"Training failed at epoch {epoch+1}: {e}")
                print("Checkpoint saved. You can resume training from this point.")
            raise


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
        print(f"Rank: {rank}")
    
    # Generate dummy data
    if rank == 0:
        print("Generating dummy data...")
    X_train, y_train = generate_dummy_data(num_samples=200, image_size=28)
    X_test, y_test = generate_dummy_data(num_samples=50, image_size=28)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
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
        
        auto_wrap_policy = size_based_auto_wrap_policy
        
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=device,
        )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Checkpoint directory - should be on shared filesystem for multi-node
    # Examples:
    #   - Local: 'checkpoints' (single node only)
    #   - NFS: '/mnt/shared/checkpoints'
    #   - Lustre: '/lustre/project/checkpoints'
    #   - S3: Use a library like s3fs
    checkpoint_dir = os.environ.get('CHECKPOINT_DIR', 'checkpoints')
    
    if rank == 0:
        print(f"Checkpoint directory: {checkpoint_dir}")
        print("Checking for existing checkpoints...")
    
    # Try to load checkpoint
    start_epoch = 0
    if os.path.exists(checkpoint_dir):
        start_epoch, last_loss = load_checkpoint(model, optimizer, checkpoint_dir)
        if start_epoch > 0 and rank == 0:
            print(f"Resuming from epoch {start_epoch}, last loss: {last_loss:.4f}")
    
    # Train
    if rank == 0:
        print("Training...")
    
    try:
        train(model, train_loader, criterion, optimizer, device, rank, 
              start_epoch=start_epoch, epochs=5, checkpoint_dir=checkpoint_dir)
    except Exception as e:
        if rank == 0:
            print(f"Training interrupted: {e}")
            print("You can restart training and it will resume from the last checkpoint.")
        cleanup_distributed()
        return
    
    # Evaluate
    if rank == 0:
        print("\nEvaluating...")
    accuracy = evaluate(model, test_loader, device, rank)
    if rank == 0:
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Save final model
    if rank == 0:
        if world_size > 1:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = model.state_dict()
                torch.save(state_dict, 'simple_cnn_fsdp_final.pth')
        else:
            torch.save(model.state_dict(), 'simple_cnn_fsdp_final.pth')
        
        print("\nFinal model saved to simple_cnn_fsdp_final.pth")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
