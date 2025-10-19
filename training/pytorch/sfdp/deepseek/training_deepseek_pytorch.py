import numpy as np
import urllib.request
import os
import sys
import tiktoken

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../model/python/model/moe/deepseek/pytorch'))
from deepseek_v3_pytorch_fsdp import DeepSeekV3Model, Block

DEEP_SEEK_CFG = {
    "vocab_size": 50257,
    "emb_dim": 64, 
    "n_layers": 1,        
    "batch": 2,
    "num_heads": 4,       # 4 heads (64/4 = 16 head_dim)
    "seq_len": 8,         # Short sequence
    "context_length": 8,  # Added for compatibility
    "num_experts": 4,     # 4 experts
    "top_k": 2,           # Top-2 routing
    "expert_dim": 128     # Small expert hidden dim
}

class DatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def create_dataloader_fsdp(txt, batch_size, max_length, stride, rank, world_size):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = DatasetV1(txt, tokenizer, max_length, stride)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return dataloader

def train_fsdp(rank, world_size):
    setup(rank, world_size)
    
    # Load data
    file_path = "../../data/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    
    # Create model
    torch.manual_seed(123)
    model = DeepSeekV3Model(DEEP_SEEK_CFG)
    
    # FSDP auto wrap policy for transformer blocks
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block}
    )
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        # Keeps parameters on GPU (not offloading to CPU). Setting to True would move unused parameters to CPU to save GPU memory
        cpu_offload=CPUOffload(offload_params=False),
        device_id=rank,
        mixed_precision=None,
    )
    
    # Create dataloader
    train_loader = create_dataloader_fsdp(
        train_data, 
        batch_size=2,
        max_length=DEEP_SEEK_CFG["context_length"],
        stride=DEEP_SEEK_CFG["context_length"], 
        rank=rank, 
        world_size=world_size
    )

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=0.0004, weight_decay=0.1)
    
    # Training loop
    fsdp_model.train()
    for epoch in range(10):
        train_loader.sampler.set_epoch(epoch)
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, fsdp_model, rank)
            loss.backward()
            optimizer.step()
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    cleanup()

def calc_loss_batch(input_batch, target_batch, model, device_id):
    input_batch = input_batch.to(device_id)
    target_batch = target_batch.to(device_id)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f'Number of GPUs {world_size}')
    mp.spawn(train_fsdp, args=(world_size,), nprocs=world_size, join=True)