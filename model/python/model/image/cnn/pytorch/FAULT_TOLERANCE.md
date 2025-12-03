# Fault Tolerance and Elastic Training

## Overview

The elastic training implementation provides fault tolerance through:
1. **Checkpointing** - Save/restore training state
2. **Elastic Training** - Handle dynamic node membership
3. **Health Checks** - Detect node failures
4. **Automatic Recovery** - Resume from checkpoints

---

## Fault Tolerance Features

### 1. **Checkpointing**

Saves training state after each epoch:
- Model weights
- Optimizer state
- Current epoch
- Loss value

```python
# Checkpoint structure
checkpoint = {
    'epoch': 5,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'loss': 0.234,
}
```

**Location:** `checkpoints/checkpoint_epoch_N.pt`

**Retention:** Keeps last 3 checkpoints to save disk space

### 2. **Automatic Resume**

On restart, automatically loads the latest checkpoint:

```python
# Checks for existing checkpoints
if os.path.exists('checkpoints'):
    start_epoch, last_loss = load_checkpoint(model, optimizer)
    # Resumes from start_epoch
```

### 3. **Health Checks**

Periodic barriers to detect node failures:

```python
# Every 10 batches, check if all nodes are alive
if batch_idx % 10 == 0:
    dist.barrier(timeout=10)  # 10 second timeout
```

If a node fails, the barrier times out and training stops gracefully.

---

## Elastic Training with torchrun

### What is Elastic Training?

Elastic training allows:
- **Dynamic node membership** - Nodes can join/leave during training
- **Automatic recovery** - Training continues with remaining nodes
- **Flexible scaling** - Start with min nodes, scale up to max nodes

### Configuration

```bash
torchrun \
    --nnodes=2:8 \          # Min 2 nodes, max 8 nodes
    --nproc_per_node=4 \    # 4 GPUs per node
    --rdzv_id=job123 \      # Unique job ID
    --rdzv_backend=c10d \   # Rendezvous backend
    --rdzv_endpoint=master:29500 \  # Master node address
    --max_restarts=3 \      # Max restart attempts
    --monitor_interval=5 \  # Health check interval (seconds)
    train_pytorch_fsdp_elastic.py
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--nnodes` | Min:Max nodes | `1:4` (1-4 nodes) |
| `--nproc_per_node` | GPUs per node | `4` |
| `--rdzv_id` | Unique job ID | `job123` |
| `--rdzv_backend` | Rendezvous method | `c10d`, `etcd` |
| `--rdzv_endpoint` | Master address | `master:29500` |
| `--max_restarts` | Restart attempts | `3` |
| `--monitor_interval` | Health check (sec) | `5` |

---

## Failure Scenarios

### Scenario 1: Single Node Failure (Multi-node setup)

**What happens:**
1. Node fails during training
2. Health check detects failure (barrier timeout)
3. Training stops, checkpoint saved
4. torchrun detects failure
5. Remaining nodes re-rendezvous
6. Training resumes from checkpoint with fewer nodes

**Example:**
```
Initial: 4 nodes (16 GPUs total)
Node 2 fails
Remaining: 3 nodes (12 GPUs)
Training continues with 12 GPUs
```

### Scenario 2: Complete Job Failure

**What happens:**
1. All nodes fail (power outage, cluster maintenance)
2. Checkpoint exists from last epoch
3. Restart job manually or via scheduler
4. Training resumes from checkpoint

**Example:**
```bash
# Initial run
torchrun --nnodes=4 --nproc_per_node=4 train_pytorch_fsdp_elastic.py
# Fails at epoch 7

# Restart (automatically resumes from epoch 7)
torchrun --nnodes=4 --nproc_per_node=4 train_pytorch_fsdp_elastic.py
```

### Scenario 3: Node Joins During Training

**What happens:**
1. Training running with N nodes
2. New node becomes available
3. torchrun detects new node
4. All nodes re-rendezvous
5. Training continues with N+1 nodes

**Note:** Requires elastic training mode (`--nnodes=min:max`)

---

## Manual Recovery

### Check Existing Checkpoints

```bash
ls checkpoints/
# checkpoint_epoch_1.pt
# checkpoint_epoch_2.pt
# checkpoint_epoch_3.pt
```

### Resume from Specific Checkpoint

```python
# Modify train_pytorch_fsdp_elastic.py
checkpoint = torch.load('checkpoints/checkpoint_epoch_5.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

### Delete Corrupted Checkpoint

```bash
# If a checkpoint is corrupted
rm checkpoints/checkpoint_epoch_7.pt
# Training will resume from epoch 6
```

---

## Best Practices

### 1. **Checkpoint Frequency**

```python
# Current: Every epoch
save_checkpoint(model, optimizer, epoch, loss)

# For long epochs, checkpoint every N batches:
if batch_idx % 1000 == 0:
    save_checkpoint(model, optimizer, epoch, loss, 
                   checkpoint_dir=f'checkpoints/batch_{batch_idx}')
```

### 2. **Checkpoint Storage**

**Critical for Multi-Node Training:**

Checkpoints MUST be on a shared filesystem accessible by all nodes:

```bash
# Set shared checkpoint directory
export CHECKPOINT_DIR="/mnt/nfs/shared/checkpoints"

# Or in your job script
torchrun --nproc_per_node=4 train_pytorch_fsdp_elastic.py
```

**Shared Filesystem Options:**

| Filesystem | Use Case | Setup |
|------------|----------|-------|
| **NFS** | Small clusters | Mount NFS share on all nodes |
| **Lustre** | HPC clusters | High-performance parallel filesystem |
| **GPFS** | IBM clusters | Parallel filesystem |
| **EFS** (AWS) | Cloud | Elastic File System |
| **Filestore** (GCP) | Cloud | Managed NFS |
| **Azure Files** | Cloud | SMB/NFS shares |

**Example Setup:**

```bash
# On all nodes, mount shared NFS
sudo mount -t nfs master:/shared /mnt/shared

# Verify all nodes can access
ls /mnt/shared  # Should work on all nodes

# Set checkpoint directory
export CHECKPOINT_DIR="/mnt/shared/checkpoints"
```

**Why Shared Storage is Required:**

```
Node 1: Saves checkpoint to /local/checkpoints/  ❌
Node 2: Looks for checkpoint in /local/checkpoints/  ❌ Not found!

Node 1: Saves checkpoint to /shared/checkpoints/  ✅
Node 2: Looks for checkpoint in /shared/checkpoints/  ✅ Found!
```

**Without Shared Storage:**
- Each node saves to its own local disk
- Other nodes can't access checkpoints
- Recovery fails on different nodes

**Alternatives for Cloud:**

```python
# Use cloud storage (requires additional libraries)
import s3fs

# S3 example
checkpoint_dir = "s3://my-bucket/checkpoints"
fs = s3fs.S3FileSystem()

# GCS example  
checkpoint_dir = "gs://my-bucket/checkpoints"
```

### 3. **Health Check Tuning**

```python
# Adjust timeout based on network latency
dist.barrier(timeout=30)  # 30 seconds for slow networks

# Adjust frequency based on batch time
if batch_idx % 50 == 0:  # Less frequent for fast batches
    dist.barrier()
```

### 4. **Monitoring**

```python
# Add logging
import logging
logging.basicConfig(level=logging.INFO)

if rank == 0:
    logging.info(f"Epoch {epoch}, Loss: {loss}")
    logging.info(f"Checkpoint saved: {checkpoint_path}")
```

---

## Limitations

### Current Implementation

1. **No mid-epoch recovery** - Restarts from last completed epoch
2. **No data loader state** - Re-iterates from start of epoch
3. **No gradient accumulation state** - Resets on restart
4. **Fixed world size** - Doesn't handle dynamic node changes mid-training

### Advanced Features (Not Implemented)

1. **Stateful DataLoader** - Resume from exact batch
2. **Gradient checkpointing** - Save intermediate activations
3. **Async checkpointing** - Non-blocking saves
4. **Cloud-native storage** - Direct S3/GCS integration

---

## Comparison

| Feature | Basic FSDP | Elastic FSDP | Production Systems |
|---------|-----------|--------------|-------------------|
| Checkpointing | ❌ | ✅ | ✅ |
| Auto-resume | ❌ | ✅ | ✅ |
| Health checks | ❌ | ✅ | ✅ |
| Dynamic nodes | ❌ | ✅ (with torchrun) | ✅ |
| Mid-epoch recovery | ❌ | ❌ | ✅ |
| Async checkpointing | ❌ | ❌ | ✅ |

---

## Testing Fault Tolerance

### Simulate Node Failure

```bash
# Terminal 1: Start training
torchrun --nproc_per_node=2 train_pytorch_fsdp_elastic.py

# Terminal 2: Kill one process during training
pkill -9 -f train_pytorch_fsdp_elastic.py

# Observe: Training stops, checkpoint saved
# Restart: Training resumes from checkpoint
```

### Test Checkpoint Recovery

```bash
# Run for 2 epochs
python train_pytorch_fsdp_elastic.py  # Stops after epoch 2

# Verify checkpoint exists
ls checkpoints/checkpoint_epoch_2.pt

# Resume training
python train_pytorch_fsdp_elastic.py  # Starts from epoch 3
```

---

## Summary

The elastic training implementation provides:

✅ **Automatic checkpointing** after each epoch
✅ **Automatic recovery** from latest checkpoint
✅ **Health monitoring** to detect failures
✅ **Graceful shutdown** on errors
✅ **Compatible with torchrun** for elastic training

For production use, consider:
- More frequent checkpointing (per N batches)
- Shared/cloud storage for checkpoints
- Monitoring and alerting
- Advanced recovery strategies
