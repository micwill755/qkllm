"""
DDP training with custom communication library (no MPI).
Uses raw TCP sockets for all communication.
"""
import numpy as np
import os
from simple_comm import init_communicator
from convolutional_numpy import Conv2D, MaxPool2D, ReLU, Flatten, Linear


class SimpleCNN:
    """Simple CNN with Conv -> ReLU -> MaxPool -> Flatten -> FC."""
    
    def __init__(self):
        self.layers = [
            Conv2D(1, 8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Conv2D(8, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Flatten(),
            Linear(7 * 7 * 16, 10)
        ]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def get_parameters(self):
        """Get all trainable parameters."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params.append(('weights', layer))
            if hasattr(layer, 'bias'):
                params.append(('bias', layer))
        return params


class DDPWrapper:
    """Distributed Data Parallel wrapper using custom communicator."""
    
    def __init__(self, model, comm):
        self.model = model
        self.comm = comm
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        
        # Broadcast initial weights from rank 0 to all processes
        self._broadcast_parameters()
    
    def _broadcast_parameters(self):
        """Broadcast model parameters from rank 0 to all processes."""
        params = self.model.get_parameters()
        
        # Flatten all parameters into a single buffer
        param_list = []
        param_shapes = []
        param_info = []
        
        for param_name, layer in params:
            param = getattr(layer, param_name)
            param_list.append(param.flatten())
            param_shapes.append(param.shape)
            param_info.append((param_name, layer))
        
        if param_list:
            param_buffer = np.concatenate(param_list)
            
            # Broadcast using our custom communicator
            self.comm.broadcast(param_buffer, root=0)
            
            # Unflatten back to individual parameters
            offset = 0
            for (param_name, layer), shape in zip(param_info, param_shapes):
                param = getattr(layer, param_name)
                size = param.size
                param.flat[:] = param_buffer[offset:offset + size]
                offset += size
    
    def _allreduce_gradients(self):
        """All-reduce gradients across all processes (average)."""
        params = self.model.get_parameters()
        
        # Flatten all gradients into a single buffer
        grad_list = []
        grad_shapes = []
        grad_info = []
        
        for param_name, layer in params:
            grad_name = f'grad_{param_name}'
            if hasattr(layer, grad_name):
                grad = getattr(layer, grad_name)
                grad_list.append(grad.flatten())
                grad_shapes.append(grad.shape)
                grad_info.append((grad_name, layer))
        
        if grad_list:
            grad_buffer = np.concatenate(grad_list)
            grad_sum_buffer = np.zeros_like(grad_buffer)
            
            # All-reduce using our custom communicator
            # Use ring all-reduce for efficiency
            self.comm.allreduce_ring(grad_buffer, grad_sum_buffer, op='sum')
            
            # Average across processes
            grad_avg_buffer = grad_sum_buffer / self.world_size
            
            # Unflatten back to individual gradients
            offset = 0
            for (grad_name, layer), shape in zip(grad_info, grad_shapes):
                grad = getattr(layer, grad_name)
                size = grad.size
                grad.flat[:] = grad_avg_buffer[offset:offset + size]
                offset += size
    
    def forward(self, x):
        return self.model.forward(x)
    
    def backward(self, grad):
        result = self.model.backward(grad)
        # Synchronize gradients after backward pass
        self._allreduce_gradients()
        return result
    
    def get_parameters(self):
        return self.model.get_parameters()


class DistributedSampler:
    """Sampler that partitions data across processes."""
    
    def __init__(self, num_samples, rank, world_size, shuffle=True, seed=42):
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        self.num_samples_per_rank = num_samples // world_size
        self.total_size = self.num_samples_per_rank * world_size
    
    def set_epoch(self, epoch):
        """Set epoch for shuffling."""
        self.epoch = epoch
    
    def get_indices(self):
        """Get indices for this rank."""
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = rng.permutation(self.num_samples)
        else:
            indices = np.arange(self.num_samples)
        
        indices = indices[:self.total_size]
        
        start_idx = self.rank * self.num_samples_per_rank
        end_idx = start_idx + self.num_samples_per_rank
        
        return indices[start_idx:end_idx]


class SGDOptimizer:
    """SGD optimizer with momentum."""
    
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        
        self.velocity = {}
        for param_name, layer in parameters:
            param = getattr(layer, param_name)
            self.velocity[(id(layer), param_name)] = np.zeros_like(param)
    
    def step(self):
        """Update parameters."""
        for param_name, layer in self.parameters:
            grad_name = f'grad_{param_name}'
            if hasattr(layer, grad_name):
                param = getattr(layer, param_name)
                grad = getattr(layer, grad_name)
                
                key = (id(layer), param_name)
                self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
                param += self.velocity[key]
    
    def zero_grad(self):
        """Zero out gradients."""
        for param_name, layer in self.parameters:
            grad_name = f'grad_{param_name}'
            if hasattr(layer, grad_name):
                grad = getattr(layer, grad_name)
                grad.fill(0)


def softmax(x):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(predictions, targets):
    """Cross entropy loss."""
    batch_size = predictions.shape[0]
    probs = softmax(predictions)
    
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(batch_size), targets] = 1
    
    loss = -np.sum(one_hot * np.log(probs + 1e-8)) / batch_size
    grad = (probs - one_hot) / batch_size
    
    return loss, grad


def generate_dummy_data(num_samples=100, image_size=28, seed=42):
    """Generate dummy data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(num_samples, 1, image_size, image_size).astype(np.float32)
    y = rng.randint(0, 10, num_samples)
    return X, y


def train_epoch(model, X_train, y_train, optimizer, sampler, batch_size, comm):
    """Train one epoch."""
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    
    indices = sampler.get_indices()
    num_samples = len(indices)
    
    epoch_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_X = X_train[batch_indices]
        batch_y = y_train[batch_indices]
        
        optimizer.zero_grad()
        
        outputs = model.forward(batch_X)
        loss, grad_output = cross_entropy_loss(outputs, batch_y)
        
        model.backward(grad_output)
        optimizer.step()
        
        epoch_loss += loss
        predictions = np.argmax(outputs, axis=1)
        correct += np.sum(predictions == batch_y)
        total += len(batch_y)
        num_batches += 1
    
    # Aggregate metrics across all processes
    local_metrics = np.array([epoch_loss, correct, total, num_batches], dtype=np.float64)
    global_metrics = np.zeros_like(local_metrics)
    
    # Use our custom all-reduce
    comm.allreduce(local_metrics, global_metrics, op='sum')
    
    avg_loss = global_metrics[0] / global_metrics[3]
    accuracy = 100 * global_metrics[1] / global_metrics[2]
    
    return avg_loss, accuracy


def evaluate(model, X_test, y_test, batch_size, comm):
    """Evaluate model."""
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    
    num_samples = len(X_test)
    samples_per_rank = num_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank if rank < world_size - 1 else num_samples
    
    X_local = X_test[start_idx:end_idx]
    y_local = y_test[start_idx:end_idx]
    
    correct = 0
    total = 0
    
    for i in range(0, len(X_local), batch_size):
        batch_X = X_local[i:i + batch_size]
        batch_y = y_local[i:i + batch_size]
        
        outputs = model.forward(batch_X)
        predictions = np.argmax(outputs, axis=1)
        
        correct += np.sum(predictions == batch_y)
        total += len(batch_y)
    
    local_metrics = np.array([correct, total], dtype=np.float64)
    global_metrics = np.zeros_like(local_metrics)
    comm.allreduce(local_metrics, global_metrics, op='sum')
    
    accuracy = 100 * global_metrics[0] / global_metrics[1]
    return accuracy


def main():
    # Initialize our custom communicator
    comm = init_communicator()
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    
    if rank == 0:
        print(f"DDP Training with Custom Communication (No MPI)")
        print(f"World size: {world_size}")
    
    # Generate data
    if rank == 0:
        print("Generating dummy data...")
    X_train, y_train = generate_dummy_data(num_samples=200, image_size=28, seed=42)
    X_test, y_test = generate_dummy_data(num_samples=50, image_size=28, seed=123)
    
    # Create model
    if rank == 0:
        print("Creating model...")
    model = SimpleCNN()
    
    # Wrap with DDP
    if rank == 0:
        print("Wrapping model with DDP...")
    ddp_model = DDPWrapper(model, comm)
    
    # Create optimizer
    optimizer = SGDOptimizer(ddp_model.get_parameters(), lr=0.01, momentum=0.9)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        num_samples=len(X_train),
        rank=rank,
        world_size=world_size,
        shuffle=True,
        seed=42
    )
    
    # Training loop
    epochs = 5
    batch_size = 16
    
    if rank == 0:
        print("Training...")
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        
        avg_loss, accuracy = train_epoch(
            ddp_model, X_train, y_train, optimizer, sampler, batch_size, comm
        )
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Evaluate
    if rank == 0:
        print("\nEvaluating...")
    accuracy = evaluate(ddp_model, X_test, y_test, batch_size, comm)
    if rank == 0:
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Cleanup
    comm.cleanup()
    
    if rank == 0:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
