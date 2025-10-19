import math
import sys
import os

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import tiktoken

# Define distributed flag
distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

# Mock SwiGLU for testing
class SwiGLU(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.linear = nn.Linear(dim1, dim2)
    def forward(self, x):
        return torch.relu(self.linear(x))

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)

class ROPE (nn.Module):
    def __init__(self, head_dim, max_seq_length=2048, base=10000):
        super().__init__()
        i = torch.arange(head_dim // 2, dtype=torch.float32)
        self.frequencies = 1 / (base ** (2 * i / head_dim))
        self.positions = torch.arange(max_seq_length, dtype=torch.float32)
        self.angles = torch.outer(self.positions, self.frequencies)

    def forward(self, x):
        # x shape: (batch, num_heads, seq_len, head_dim)
        batch, num_heads, seq_len, head_dim = x.shape
        x = x.view(batch, num_heads, seq_len, head_dim // 2, 2)
        
        for pos in range(seq_len):
            for pair_idx in range(head_dim // 2):
                angle = self.angles[pos, pair_idx]
                X = x[:, :, pos, pair_idx, 0]
                Y = x[:, :, pos, pair_idx, 1]
                x[:, :, pos, pair_idx, 0] = X * torch.cos(angle) - Y * torch.sin(angle)
                x[:, :, pos, pair_idx, 1] = X * torch.sin(angle) + Y * torch.cos(angle)
        
        return x.view(batch, num_heads, seq_len, head_dim)

class Expert(nn.Module):
    def __init__(self, emb_dim, expert_dim):
        super().__init__()
        self.layer1 = nn.Linear(emb_dim, expert_dim)
        self.swiGLU = SwiGLU(expert_dim, expert_dim) 
        self.layer2 = nn.Linear(expert_dim, emb_dim)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.swiGLU.forward(x)
        x = self.layer2.forward(x)

        return x

class MoE(nn.Module):
    def __init__(self, emb_dim, num_experts, top_k, expert_dim, load_balance_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(emb_dim, num_experts)

        self.distributed = distributed
        self.world_size = dist.get_world_size() if distributed else 1
        self.rank = dist.get_rank() if distributed else 0
        
        # Validate expert distribution
        if num_experts % self.world_size != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by world_size ({self.world_size})")
        
        # Calculate which experts this GPU owns
        self.experts_per_gpu = num_experts // self.world_size
        self.local_expert_start = self.rank * self.experts_per_gpu
        self.local_expert_end = self.local_expert_start + self.experts_per_gpu
        
        # Only create local experts
        self.experts = nn.ModuleList([
            Expert(emb_dim, expert_dim) 
            for _ in range(self.experts_per_gpu)
        ])

    def is_expert_local(self, expert_idx):
        """Check if expert_idx is on this GPU"""
        if not self.distributed:
            return True
        return self.local_expert_start <= expert_idx < self.local_expert_end

    def batch_remote_expert_calls(self, tokens_by_expert, expert_indices):
        """More efficient: batch all remote calls together"""
        world_size = self.world_size
        
        # Prepare send buffers for each GPU
        send_data = [[] for _ in range(world_size)]
        for expert_idx, tokens in tokens_by_expert.items():
            target_gpu = expert_idx // self.experts_per_gpu
            send_data[target_gpu].extend(tokens)
        
        # TODO: All-to-all exchange (simplified - actual implementation needs proper tensor serialization)
        received_data = send_data  # Placeholder for actual distributed communication
        
        # Process local experts
        results = {}
        for expert_idx, tokens in tokens_by_expert.items():
            if self.is_expert_local(expert_idx):
                local_idx = expert_idx - self.local_expert_start
                token_tensor = torch.stack(tokens)
                results[expert_idx] = self.experts[local_idx](token_tensor)
        
        # All-to-all return results
        return_data = [[] for _ in range(world_size)]
        for expert_idx, result in results.items():
            target_gpu = expert_idx // self.experts_per_gpu
            return_data[target_gpu].append((expert_idx, result))
        
        final_results = [[] for _ in range(world_size)]
        dist.all_to_all(final_results, return_data)
        
        # Flatten results
        all_results = {}
        for gpu_results in final_results:
            for expert_idx, result in gpu_results:
                all_results[expert_idx] = result
        
        return all_results


    def forward(self, x):
        # x shape: (batch, seq_len, emb_dim)
        batch_size, seq_len, emb_dim = x.shape

        # step 1 router computation
        expert_scores = self.router(x)

        # step 2 - softmax scores to probabilities
        expert_probs = torch.softmax(expert_scores, dim=-1)

        # step 3 - top k take the top 2 highest probabiliti experts for every token
        top_k_probs, top_k_indices = torch.topk(expert_probs, k=self.top_k, dim=-1)

        # step 4 Expert routing and weighted combination
        output = torch.zeros_like(x)

        if self.distributed:
            # Collect all tokens by expert
            tokens_by_expert = {}
            token_positions = {}  # Track where each token goes
            
            for b in range(batch_size):
                for s in range(seq_len):
                    for k in range(self.top_k):
                        expert_idx = top_k_indices[b, s, k].item()
                        token_embedding = x[b, s, :]
                        
                        if expert_idx not in tokens_by_expert:
                            tokens_by_expert[expert_idx] = []
                            token_positions[expert_idx] = []
                        
                        tokens_by_expert[expert_idx].append(token_embedding)
                        token_positions[expert_idx].append((b, s, k))
            
            # Batch process all expert calls
            expert_results = self.batch_remote_expert_calls(tokens_by_expert, list(tokens_by_expert.keys()))
            
            # Reconstruct output using results
            output = torch.zeros_like(x)
            for expert_idx, results in expert_results.items():
                positions = token_positions[expert_idx]
                for i, (b, s, k) in enumerate(positions):
                    prob = top_k_probs[b, s, k]
                    output[b, s, :] += prob * results[i]
        else:
            # Original token-by-token processing for non-distributed
            output = torch.zeros_like(x)
            for b in range(batch_size):
                for s in range(seq_len):
                    token_embedding = x[b, s, :]
                    weighted_sum = torch.zeros(emb_dim, device=x.device, dtype=x.dtype)
                    
                    for k in range(self.top_k):
                        expert_idx = top_k_indices[b, s, k].item()
                        prob = top_k_probs[b, s, k]
                        expert_output = self.experts[expert_idx](token_embedding)
                        weighted_sum += prob * expert_output
                        
                    output[b, s, :] = weighted_sum

        return output
    
class MultiHeadLatentAttention (nn.Module):
    # emb_dim = also known as d_model, is the size of the token embeddings eg. 768
    def __init__(self, emb_dim, num_heads, max_seq_length, latent_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.latent_dim = latent_dim

        self.rope = ROPE(head_dim=self.head_dim, max_seq_length=max_seq_length)

        self.latent_tokens = nn.Linear(max_seq_length * emb_dim, latent_dim * emb_dim)
        self.query_W = nn.Linear(emb_dim, emb_dim)
        self.key_W = nn.Linear(emb_dim, emb_dim)
        self.value_W = nn.Linear(emb_dim, emb_dim)
        self.output_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        
        x_flat = x.view(batch_size, -1)  # (batch, seq_len * emb_dim)
        latent_compressed = self.latent_tokens(x_flat)  # (batch, latent_dim * emb_dim)
        latent = latent_compressed.view(batch_size, self.latent_dim, emb_dim) # (batch, latent_dim, emb_dim)

        # step 1 - Generate Q, K, V (CROSS-ATTENTION)
        query = self.query_W(x)        # (batch, seq_len, emb_dim)
        key = self.key_W(latent)       # (batch, latent_dim, emb_dim)
        value = self.value_W(latent)   # (batch, latent_dim, emb_dim)

        # step 2: Reshape for multi-head attention FIRST
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, seq_len, head_dim)
        key = key.view(batch_size, self.latent_dim, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, latent_dim, head_dim)
        value = value.view(batch_size, self.latent_dim, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, latent_dim, head_dim)
        
        # step 3: Apply RoPE AFTER reshaping 
        query_r = self.rope(query)     # (batch, num_heads, seq_len, head_dim)
        key_r = key                    # No RoPE for latent tokens
            
        # step 4: attention computation 
        attn_scores = query_r @ key_r.transpose(-2, -1) / math.sqrt(self.head_dim)
        # (batch, num_heads, seq_len, latent_dim)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ value  # (batch, num_heads, seq_len, head_dim)

        # NEW: Combine heads back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        
        return self.output_proj(attn_output)

class Block(nn.Module):
    def __init__(self, batch, emb_dim, num_heads, seq_len, num_experts, top_k, expert_dim):
        super().__init__()
        self.batch = batch
        self.emb_dim = emb_dim
        self.seq_len = seq_len

        self.rmsNorm1 = RMSNorm(emb_dim)
        self.MHLA = MultiHeadLatentAttention(emb_dim, num_heads, seq_len)
        self.rmsNorm2 = RMSNorm(emb_dim)
        self.moE = MoE(emb_dim, num_experts, top_k, expert_dim)

    def forward(self, x):
        # First sub-layer: Attention with residual
        attn_output = self.MHLA(self.rmsNorm1(x))
        x = x + attn_output
        # Second sub-layer: MoE with residual  
        moe_output = self.moE(self.rmsNorm2(x))
        x = x + moe_output
        return x

class DeepSeekV3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.blocks = nn.ModuleList([
            Block(cfg["batch"], cfg["emb_dim"], cfg["num_heads"], 
                  cfg["seq_len"], cfg["num_experts"], cfg["top_k"], 
                  cfg["expert_dim"]) for _ in range(cfg["n_layers"])
        ])
        self.finalRMSNorm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        # Embedding lookup
        x = self.tok_emb(x)  # (batch, seq_len, emb_dim)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final norm and output
        x = self.finalRMSNorm(x)
        logits = self.out_head(x)  # (1, seq_len, vocab_size)
        
        return logits  # (batch, seq_len, vocab_size)

tokenizer = tiktoken.get_encoding("gpt2")

cfg = {
    "vocab_size": 50257,
    "emb_dim": 768, 
    "n_layers": 61,
    "batch": 1,
    "num_heads": 12,
    "seq_len": 1024,
    "num_experts": 8,
    "top_k": 2,
    "expert_dim": 2048
}

# Unit Tests
def test_deepseek_model():
    print("Testing DeepSeek V3 Model...")
    
    # Small config for fast testing
    test_cfg = {
        "vocab_size": 50257,  # Full GPT-2 vocab to handle real tokens
        "emb_dim": 64,        # Small embedding
        "n_layers": 1,        # Only 1 layer
        "batch": 1,
        "num_heads": 4,       # 4 heads (64/4 = 16 head_dim)
        "seq_len": 8,         # Short sequence
        "num_experts": 4,     # 4 experts
        "top_k": 2,           # Top-2 routing
        "expert_dim": 128     # Small expert hidden dim
    }
    
    try:
        # Create model
        print("Creating model...")
        test_model = DeepSeekV3Model(test_cfg)
        
        # Test input
        test_text = "Hello world, this is a test."
        test_input = torch.tensor(tokenizer.encode(test_text)[:test_cfg["seq_len"]]).unsqueeze(0)
        print(f"Input tokens: {test_input}")
        
        # Forward pass
        print("Running forward pass...")
        logits = test_model(test_input)
        
        # Check output shape
        expected_shape = (1, test_cfg["seq_len"], test_cfg["vocab_size"])
        
        if logits.shape == expected_shape:
            print("✅ Model test PASSED!")
            print(f"Output shape: {logits.shape}")
            print(f"Sample logits: {logits[0, :5]}")  # First token, first 5 vocab scores
            return True
        else:
            print(f"❌ Wrong output shape: {logits.shape}, expected {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ Model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run tests
if __name__ == "__main__":
    print("Running DeepSeek V3 Tests...\n")
    
    model_test = test_deepseek_model()
    
    if model_test:
        print("\n🎉 All tests passed! Your DeepSeek V3 model is working!")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")