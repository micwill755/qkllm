import numpy as np
import pandas as pd
import os 
import requests
import tiktoken
import torch
import sys
import os
import math

from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.python.model.moe.deepseek.pytorch.deepseek_v3_pytorch import DeepSeekV3Model, generate_text_simple, text_to_tokens, token_ids_to_text

# Q3 implement a forward pass for an llm using text data

test_cfg = {
    "vocab_size": 50257,  # Full GPT-2 vocab to handle real tokens
    "emb_dim": 64,        # Small embedding
    "n_layers": 1,        # Only 1 layer
    "batch": 1,
    "num_heads": 4,       # 4 heads (64/4 = 16 head_dim)
    "seq_len": 8,         # Short sequence
    "num_experts": 4,     # 4 experts
    "top_k": 2,           # Top-2 routing
    "expert_dim": 128,     # Small expert hidden dim
    "context_length": 256
}

# load data
input_file = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file):
    #url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(requests.get(url).text)

with open(input_file, 'r', encoding='utf-8') as f:
    data = f.read()

n = len(data)

tokenizer = tiktoken.get_encoding('gpt2')
token_ids = tokenizer.encode(data)
print('Characters: ', n)
print('Tokens: ', len(token_ids))

train_ratio = 0.9
split_idx = int(n * train_ratio)
train_data = data[:split_idx]
test_data = data[split_idx:]

train_tokens = tokenizer.encode(train_data)
test_tokens = tokenizer.encode(test_data)

input_ids = []
target_ids = []
max_length = test_cfg["seq_len"]

for i in range(len(train_tokens) - max_length):  # Ensure we don't go past end
    input_chunk = train_tokens[i: i + max_length]
    target_chunk = train_tokens[i + 1: i + 1 + max_length]
    if len(input_chunk) == max_length and len(target_chunk) == max_length:
        input_ids.append(torch.tensor(input_chunk))
        target_ids.append(torch.tensor(target_chunk))

print('Input data:', len(input_ids))
print('Target data:', len(target_ids))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = DeepSeekV3Model(test_cfg).to(device)

# multiple classes
def categorical_cross_entropy(logits, targets):
    # Softmax: convert logits to probabilities
    exp_logits = torch.exp(logits - torch.max(logits, dim=-1, keepdim=True)[0])
    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    
    # Get probability of correct token for each position
    batch_size, seq_len = targets.shape
    flat_targets = targets.view(-1)
    flat_probs = probs.view(-1, probs.size(-1))
    
    # Extract probabilities for target tokens
    target_probs = flat_probs[torch.arange(len(flat_targets)), flat_targets]

    # Cross entropy: -mean(log(p_correct))
    return -torch.mean(torch.log(target_probs + 1e-8))

def train_batched(train_loader, optimizer, epochs=100):
    for epoch in range(epochs):
        total_loss = 0
        for batch_inputs, batch_targets in train_loader:
            # Move data to GPU
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = categorical_cross_entropy(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')
        prompt = "Hello world, my"
        out = generate_text_simple(model, text_to_tokens(prompt, tokenizer).to(device), 10, test_cfg, 50256)
        output = token_ids_to_text(out, tokenizer)
        print(output)

dataset = TensorDataset(torch.stack(input_ids), torch.stack(target_ids))
train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
train_batched(train_loader, optimizer, 100)

dummy_input = torch.randint(0, test_cfg["vocab_size"], (1, test_cfg["seq_len"])).to(device)
onnx_path = "deepseek_v3.onnx"
if not os.path.exists(onnx_path):
    torch.onnx.export(model, dummy_input, onnx_path,
        input_names=['input_ids'], output_names=['logits'])