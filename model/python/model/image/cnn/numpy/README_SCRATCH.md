# DDP From Scratch - No MPI Required

This implements Distributed Data Parallel training completely from scratch using only Python sockets. No MPI, no PyTorch distributed, just raw TCP/IP communication.

## What We Built

### 1. Custom Communicator (`simple_comm.py`)

Implements the core collective operations:

- **Broadcast**: One process sends data to all others
- **Allreduce**: Combine data from all processes and distribute result
- **Barrier**: Synchronization point where all processes wait
- **Send/Recv**: Point-to-point communication

### 2. Communication Architecture

```
Process 0 (Master)
    ↓ (listens on port 29500)
    ├─ Process 1 connects
    ├─ Process 2 connects  
    └─ Process 3 connects

Then establishes peer-to-peer mesh:
    
    Process 0 ←→ Process 1
        ↕            ↕
    Process 3 ←→ Process 2
```

Each process can communicate directly with any other process.

### 3. Ring All-Reduce Algorithm

Instead of gathering everything to rank 0, we use a ring topology:

```
Step 1: Divide data into chunks
Process 0: [A0, B0, C0, D0]
Process 1: [A1, B1, C1, D1]
Process 2: [A2, B2, C2, D2]
Process 3: [A3, B3, C3, D3]

Step 2: Reduce-scatter (each process accumulates one chunk)
Round 1: 0→1, 1→2, 2→3, 3→0
Round 2: 0→1, 1→2, 2→3, 3→0
Round 3: 0→1, 1→2, 2→3, 3→0

Step 3: All-gather (distribute complete chunks)
Round 1: 0→1, 1→2, 2→3, 3→0
Round 2: 0→1, 1→2, 2→3, 3→0
Round 3: 0→1, 1→2, 2→3, 3→0

Result: All processes have [A_sum, B_sum, C_sum, D_sum]
```

This is O(N) communication instead of O(N²) for naive gather-reduce-broadcast.

## Installation

No special dependencies! Just Python and NumPy:

```bash
pip install numpy
```

## Running

### Using the launch script (recommended):

```bash
# Make script executable
chmod +x launch_ddp.sh

# Launch with 4 processes
./launch_ddp.sh 4

# Launch with 2 processes
./launch_ddp.sh 2

# Custom master address and port
./launch_ddp.sh 4 192.168.1.100 12345
```

### Manual launch:

```bash
# Terminal 1 (Rank 0)
RANK=0 WORLD_SIZE=4 MASTER_ADDR=localhost MASTER_PORT=29500 \
python train_numpy_ddp_scratch.py &

# Terminal 2 (Rank 1)
RANK=1 WORLD_SIZE=4 MASTER_ADDR=localhost MASTER_PORT=29500 \
python train_numpy_ddp_scratch.py &

# Terminal 3 (Rank 2)
RANK=2 WORLD_SIZE=4 MASTER_ADDR=localhost MASTER_PORT=29500 \
python train_numpy_ddp_scratch.py &

# Terminal 4 (Rank 3)
RANK=3 WORLD_SIZE=4 MASTER_ADDR=localhost MASTER_PORT=29500 \
python train_numpy_ddp_scratch.py &
```

## How It Works

### Connection Setup

1. **Rank 0 (Master)** starts a TCP server on `MASTER_PORT`
2. **Other ranks** connect to master and send their rank ID
3. **Master** collects connection info and broadcasts peer addresses
4. **All ranks** establish peer-to-peer connections in a mesh topology

### Broadcast Operation

```python
def broadcast(self, data, root=0):
    if self.rank == root:
        # Root sends to everyone
        for rank in range(self.world_size):
            if rank != root:
                self._send_data(self.sockets[rank], data)
    else:
        # Others receive from root
        received = self._recv_data(self.sockets[root])
        np.copyto(data, received)
```

**What happens:**
- Root serializes numpy array with pickle
- Sends size header (4 bytes) then data
- Other ranks receive and deserialize
- All ranks end up with same data

### All-Reduce Operation (Ring Algorithm)

```python
def allreduce_ring(self, send_data, recv_data, op='sum'):
    # Phase 1: Reduce-scatter
    for step in range(world_size - 1):
        # Send chunk to next rank in ring
        # Receive chunk from previous rank
        # Accumulate received chunk
    
    # Phase 2: All-gather
    for step in range(world_size - 1):
        # Send accumulated chunk to next rank
        # Receive complete chunk from previous rank
```

**Efficiency:**
- Communication: 2(N-1) steps where N = world_size
- Data transferred per process: 2(N-1)/N × data_size
- Bandwidth optimal: doesn't bottleneck on any single process

### Gradient Synchronization

```python
# In backward pass:
def backward(self, grad):
    result = self.model.backward(grad)
    
    # Flatten all gradients
    grad_buffer = np.concatenate([g.flatten() for g in grads])
    
    # Ring all-reduce (THIS IS WHERE PROCESSES WAIT)
    self.comm.allreduce_ring(grad_buffer, grad_sum_buffer, op='sum')
    
    # Average
    grad_avg = grad_sum_buffer / world_size
    
    # Unflatten back
    # ...
    
    return result
```

## Comparison

| Feature | MPI Version | From-Scratch Version |
|---------|-------------|---------------------|
| Dependencies | mpi4py, MPI library | Just Python stdlib |
| Communication | MPI primitives | Raw TCP sockets |
| All-Reduce | MPI.Allreduce | Custom ring algorithm |
| Broadcast | MPI.Bcast | Custom implementation |
| Setup | mpirun command | Shell script or manual |
| Portability | Requires MPI install | Works anywhere |
| Performance | Highly optimized | Good for learning |

## Performance Notes

**Advantages:**
- No external dependencies
- Easy to understand and modify
- Works on any system with Python
- Good for learning distributed algorithms

**Disadvantages:**
- Slower than optimized MPI/NCCL (no RDMA, no GPU-direct)
- Uses pickle for serialization (overhead)
- TCP sockets have higher latency than InfiniBand/RDMA
- No fault tolerance

**When to use:**
- Learning distributed training concepts
- Prototyping distributed algorithms
- Systems without MPI/NCCL
- Small-scale experiments

**When NOT to use:**
- Production training at scale
- GPU clusters with InfiniBand
- Need maximum performance

## Key Concepts Demonstrated

1. **Socket Programming**: TCP client/server, send/recv
2. **Serialization**: Converting numpy arrays to bytes
3. **Collective Operations**: Broadcast, all-reduce, barrier
4. **Ring Algorithms**: Efficient all-reduce without bottlenecks
5. **Process Coordination**: Synchronization and data exchange
6. **Distributed Training**: Data parallelism, gradient averaging

This is essentially what PyTorch's `torch.distributed` and MPI do under the hood, just with more optimizations!
