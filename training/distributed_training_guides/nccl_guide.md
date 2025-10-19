# NCCL (NVIDIA Collective Communications Library) Guide

## Setup

```bash
# Install NCCL (usually comes with CUDA toolkit)
sudo apt-get install libnccl2 libnccl-dev

# Or build from source
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build
```

## Basic NCCL C++ Example

```cpp
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
    int nDevs;
    cudaGetDeviceCount(&nDevs);
    
    // Initialize NCCL
    ncclComm_t comms[nDevs];
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    
    // Create communicators
    ncclGroupStart();
    for (int i = 0; i < nDevs; i++) {
        cudaSetDevice(i);
        ncclCommInitRank(&comms[i], nDevs, id, i);
    }
    ncclGroupEnd();
    
    // Allocate data on each GPU
    float **sendbuff = new float*[nDevs];
    float **recvbuff = new float*[nDevs];
    cudaStream_t *streams = new cudaStream_t[nDevs];
    
    size_t size = 1024 * 1024; // 1M floats
    
    for (int i = 0; i < nDevs; i++) {
        cudaSetDevice(i);
        cudaMalloc(&sendbuff[i], size * sizeof(float));
        cudaMalloc(&recvbuff[i], size * sizeof(float));
        cudaStreamCreate(&streams[i]);
        
        // Initialize data
        cudaMemset(sendbuff[i], i, size * sizeof(float));
    }
    
    // All-reduce operation
    ncclGroupStart();
    for (int i = 0; i < nDevs; i++) {
        cudaSetDevice(i);
        ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat, ncclSum, comms[i], streams[i]);
    }
    ncclGroupEnd();
    
    // Synchronize
    for (int i = 0; i < nDevs; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    
    // Cleanup
    for (int i = 0; i < nDevs; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(comms[i]);
        cudaFree(sendbuff[i]);
        cudaFree(recvbuff[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    delete[] sendbuff;
    delete[] recvbuff;
    delete[] streams;
    
    return 0;
}
```

## Compilation

```bash
nvcc -o nccl_example nccl_example.cu -lnccl
```

## Python NCCL Integration

```python
import torch
import torch.distributed as dist

# Initialize process group with NCCL backend
dist.init_process_group(backend='nccl', init_method='env://')

def all_reduce_example():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create tensor on current GPU
    tensor = torch.ones(1000, 1000).cuda() * rank
    
    # All-reduce sum
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Result should be sum of all ranks
    expected = sum(range(world_size))
    print(f"Rank {rank}: tensor[0,0] = {tensor[0,0].item()}, expected = {expected}")

def all_gather_example():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create tensor with rank-specific data
    tensor = torch.ones(100).cuda() * rank
    
    # Gather tensors from all ranks
    tensor_list = [torch.zeros(100).cuda() for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    print(f"Rank {rank}: gathered {[t[0].item() for t in tensor_list]}")

if __name__ == "__main__":
    all_reduce_example()
    all_gather_example()
    dist.destroy_process_group()
```

## NCCL Environment Variables

```bash
# Performance tuning
export NCCL_DEBUG=INFO                    # Enable debug output
export NCCL_IB_DISABLE=0                  # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=2              # GPU Direct RDMA level
export NCCL_P2P_DISABLE=0                # Enable P2P communication
export NCCL_SHM_DISABLE=0                # Enable shared memory

# Network configuration
export NCCL_SOCKET_IFNAME=eth0            # Network interface
export NCCL_IB_HCA=mlx5_0                # InfiniBand adapter
export NCCL_ALGO=Ring                     # Communication algorithm

# Topology awareness
export NCCL_TOPO_FILE=/path/to/topo.xml   # Custom topology file
export NCCL_GRAPH_FILE=/path/to/graph.xml # Communication graph
```

## Multi-Node Setup

```bash
# Node 0 (master)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0
python nccl_multinode.py

# Node 1 (worker)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=4
python nccl_multinode.py
```

## Performance Optimization

- **Bandwidth**: Use InfiniBand or high-speed Ethernet
- **Topology**: Configure NCCL topology for optimal routing
- **Algorithms**: Choose between Ring, Tree, or CollNet algorithms
- **Buffering**: Tune buffer sizes for your network
- **Compression**: Enable compression for slow networks

## Collective Operations

- `ncclAllReduce`: Reduce and broadcast result to all ranks
- `ncclBroadcast`: Send data from one rank to all others
- `ncclAllGather`: Gather data from all ranks to all ranks
- `ncclReduceScatter`: Reduce and scatter results
- `ncclReduce`: Reduce data to single rank
- `ncclSend/ncclRecv`: Point-to-point communication