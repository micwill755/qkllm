# NumPy DDP Training

This implements Distributed Data Parallel (DDP) training from scratch using NumPy and MPI.

## Key Components

### 1. DDPWrapper
- Broadcasts initial weights from rank 0 to all processes
- Synchronizes gradients via all-reduce after backward pass
- Averages gradients across all processes

### 2. DistributedSampler
- Partitions data across processes
- Each process gets a unique subset of data
- Supports shuffling with reproducible seeds per epoch

### 3. Gradient Synchronization
- After backward pass, gradients are all-reduced (summed) across processes
- Gradients are averaged by dividing by world_size
- This ensures all processes have identical gradients

### 4. Metric Aggregation
- Loss, accuracy, and other metrics are aggregated via all-reduce
- Ensures consistent reporting across all processes

## Installation

```bash
# Install MPI (if not already installed)
# On macOS:
brew install open-mpi

# On Ubuntu:
sudo apt-get install libopenmpi-dev

# Install mpi4py
pip install mpi4py numpy
```

## Running

### Single Process (for testing)
```bash
python train_numpy_ddp.py
```

### Multiple Processes (DDP)
```bash
# Run with 4 processes
mpirun -n 4 python train_numpy_ddp.py

# Run with 2 processes
mpirun -n 2 python train_numpy_ddp.py
```

## How It Works

### Comparison with PyTorch DDP

| Component | PyTorch DDP | NumPy DDP |
|-----------|-------------|-----------|
| Process Init | `torch.distributed.init_process_group()` | `MPI.COMM_WORLD` |
| Model Wrapper | `DistributedDataParallel(model)` | `DDPWrapper(model, comm)` |
| Weight Broadcast | Automatic in DDP | `comm.Bcast()` in `_broadcast_parameters()` |
| Gradient Sync | Automatic during backward | `comm.Allreduce()` in `_allreduce_gradients()` |
| Data Partitioning | `DistributedSampler` | Custom `DistributedSampler` |
| Metric Aggregation | Manual `dist.all_reduce()` | `comm.Allreduce()` |

### Key Differences from PyTorch

1. **Explicit Gradient Sync**: In PyTorch DDP, gradient synchronization happens automatically during `loss.backward()`. In our NumPy version, we explicitly call `_allreduce_gradients()` after the backward pass.

2. **MPI vs NCCL/Gloo**: PyTorch uses NCCL (GPU) or Gloo (CPU) backends. We use MPI which works on both CPU and distributed systems.

3. **Manual Parameter Broadcast**: We explicitly broadcast parameters from rank 0 to all processes at initialization.

4. **Sampler Logic**: We implement the data partitioning logic manually to ensure each process gets unique data.

## Architecture

```
Process 0: [Data Shard 0] -> Model Copy 0 -> Gradients -> All-Reduce
Process 1: [Data Shard 1] -> Model Copy 1 -> Gradients -> All-Reduce
Process 2: [Data Shard 2] -> Model Copy 2 -> Gradients -> All-Reduce
Process 3: [Data Shard 3] -> Model Copy 3 -> Gradients -> All-Reduce
                                                    ↓
                                            Averaged Gradients
                                                    ↓
                                            Update All Models
```

Each process:
1. Gets a unique shard of data
2. Runs forward pass on its shard
3. Computes gradients via backward pass
4. All-reduces gradients (sum across all processes)
5. Averages gradients by dividing by world_size
6. Updates model parameters (all models stay in sync)

## Performance Notes

- **Scaling**: With N processes, you get ~N times speedup (minus communication overhead)
- **Communication**: All-reduce is the main communication bottleneck
- **Batch Size**: Effective batch size = local_batch_size × world_size
- **Memory**: Each process holds a full copy of the model but only a shard of data
