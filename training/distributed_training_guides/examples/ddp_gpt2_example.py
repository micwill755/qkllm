#!/usr/bin/env python3
"""
DDP training example for GPT-2 model
Usage: python -m torch.distributed.launch --nproc_per_node=4 ddp_gpt2_example.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import argparse

class SimpleGPT2(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_seq_len=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4*d_model, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def create_dummy_dataset(size=10000, seq_len=512, vocab_size=50257):
    """Create dummy dataset for demonstration"""
    data = torch.randint(0, vocab_size, (size, seq_len))
    return torch.utils.data.TensorDataset(data)

def train_epoch(model, dataloader, optimizer, criterion, rank, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (data,) in enumerate(dataloader):
        data = data.to(rank)
        
        # Shift for causal language modeling
        input_ids = data[:, :-1]
        targets = data[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if rank == 0 and batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    # Setup distributed training
    rank = args.local_rank
    world_size = torch.cuda.device_count()
    setup_ddp(rank, world_size)
    
    # Create model and move to GPU
    model = SimpleGPT2()
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Create dataset and dataloader
    dataset = create_dummy_dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training loop
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        
        avg_loss = train_epoch(ddp_model, dataloader, optimizer, criterion, rank, epoch)
        
        if rank == 0:
            print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')
    
    # Save model (only on rank 0)
    if rank == 0:
        torch.save(model.state_dict(), 'gpt2_ddp_checkpoint.pt')
        print("Model saved!")
    
    cleanup_ddp()

if __name__ == "__main__":
    main()