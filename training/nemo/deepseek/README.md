# DeepSeek V3 NeMo Implementation

A production-ready implementation of DeepSeek V3 with expert parallelism using NVIDIA NeMo framework.

## Features

- ✅ **Expert Parallelism** - Distribute MoE experts across multiple GPUs
- ✅ **Multi-Head Latent Attention** - Efficient attention with compressed latent tokens
- ✅ **NeMo Integration** - Production-grade distributed training
- ✅ **Flexible Scaling** - Single GPU to multi-node clusters
- ✅ **Configuration-Driven** - YAML-based experiment management

## Quick Start

### 1. Test the Model
```bash
cd training/nemo/deepseek
python scripts/test_model.py
```

### 2. Single GPU Training
```bash
python scripts/train_deepseek.py --config deepseek_v3_base
```

### 3. Multi-GPU Training
```bash
python scripts/train_deepseek.py --config deepseek_v3_distributed
```

## Architecture

### Model Components
- **DeepSeekBlock** - Transformer block with latent attention + MoE
- **MultiHeadLatentAttention** - Cross-attention to compressed latent tokens
- **MoE** - Mixture of Experts with distributed expert parallelism
- **Expert** - Individual expert networks with SwiGLU activation

### Expert Parallelism
- Experts distributed across GPUs using deterministic partitioning
- Batch processing for efficient cross-GPU communication
- Automatic load balancing and routing optimization

## Configuration

### Base Config (`deepseek_v3_base.yaml`)
- Single GPU training
- 12 layers, 768 embedding dim
- 8 experts, top-2 routing

### Distributed Config (`deepseek_v3_distributed.yaml`)
- Multi-GPU training
- 24 layers, 1024 embedding dim
- 16 experts across 4 GPUs

## Directory Structure

```
training/nemo/deepseek/
├── models/
│   ├── deepseek_model.py      # Main model with NeMo integration
│   └── components/            # Modular architecture components
├── configs/                   # YAML configuration files
├── scripts/                   # Training and testing scripts
└── docs/                      # Documentation and guides
```

## Requirements

- PyTorch >= 1.13
- NVIDIA NeMo (optional, falls back to PyTorch Lightning)
- PyTorch Lightning
- OmegaConf

## Installation

```bash
# Install NeMo (recommended)
pip install nemo_toolkit

# Or just PyTorch Lightning (fallback)
pip install pytorch-lightning omegaconf
```

## Scaling Examples

### 2 GPUs
```yaml
trainer:
  devices: 2
model:
  expert_model_parallel_size: 2
```

### 8 GPUs
```yaml
trainer:
  devices: 8
model:
  expert_model_parallel_size: 4
  tensor_model_parallel_size: 2
```

### Multi-Node (32 GPUs)
```yaml
trainer:
  devices: 8
  num_nodes: 4
model:
  expert_model_parallel_size: 8
  tensor_model_parallel_size: 2
  pipeline_model_parallel_size: 2
```

## Expert Parallelism Details

The implementation distributes experts across GPUs using:

1. **Deterministic Partitioning** - `expert_id // experts_per_gpu`
2. **Token Batching** - Group tokens by target expert for efficiency
3. **All-to-All Communication** - Optimized cross-GPU expert calls
4. **Position Tracking** - Maintain token positions for result reconstruction

## Monitoring

The framework provides built-in monitoring for:
- Expert utilization per GPU
- Communication patterns and timing
- Memory usage and optimization
- Training metrics and convergence

## Migration from PyTorch FSDP

This implementation preserves all concepts from the original PyTorch FSDP version while adding:
- Production-grade distributed training
- Configuration-driven experiments
- Automatic scaling and optimization
- Advanced monitoring and checkpointing

See `docs/nemo_migration_guide.md` for detailed migration information.

## Contributing

1. Follow the modular architecture in `models/components/`
2. Add new configurations in `configs/`
3. Update tests in `scripts/test_model.py`
4. Document changes in `docs/`