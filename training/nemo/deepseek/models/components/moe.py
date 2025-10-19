import torch
import torch.nn as nn
import torch.distributed as dist
from .norms import SwiGLU


class Expert(nn.Module):
    """Individual expert in MoE layer"""
    
    def __init__(self, emb_dim, expert_dim):
        super().__init__()
        self.layer1 = nn.Linear(emb_dim, expert_dim)
        self.swiGLU = SwiGLU(expert_dim, expert_dim) 
        self.layer2 = nn.Linear(expert_dim, emb_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.swiGLU(x)
        x = self.layer2(x)
        return x


class MoE(nn.Module):
    """Mixture of Experts with distributed expert parallelism"""
    
    def __init__(self, emb_dim, num_experts, top_k, expert_dim, load_balance_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(emb_dim, num_experts)
        self.load_balance_weight = load_balance_weight

        # Distributed setup
        self.distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0
        
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
        """Batch process expert calls with distributed communication"""
        if not self.distributed:
            # Single GPU case - process all experts locally
            results = {}
            for expert_idx, tokens in tokens_by_expert.items():
                token_tensor = torch.stack(tokens)
                results[expert_idx] = self.experts[expert_idx](token_tensor)
            return results
        
        # Distributed case - use NeMo's optimized expert parallelism
        # This would integrate with NeMo's MoE implementation
        results = {}
        for expert_idx, tokens in tokens_by_expert.items():
            if self.is_expert_local(expert_idx):
                local_idx = expert_idx - self.local_expert_start
                token_tensor = torch.stack(tokens)
                results[expert_idx] = self.experts[local_idx](token_tensor)
        
        return results

    def forward(self, x):
        # x shape: (batch, seq_len, emb_dim)
        batch_size, seq_len, emb_dim = x.shape

        # Router computation
        expert_scores = self.router(x)
        expert_probs = torch.softmax(expert_scores, dim=-1)

        # Top-k expert selection
        top_k_probs, top_k_indices = torch.topk(expert_probs, k=self.top_k, dim=-1)

        # Expert routing and weighted combination
        output = torch.zeros_like(x)

        if self.distributed:
            # Collect all tokens by expert
            tokens_by_expert = {}
            token_positions = {}
            
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
            for expert_idx, results in expert_results.items():
                positions = token_positions[expert_idx]
                for i, (b, s, k) in enumerate(positions):
                    prob = top_k_probs[b, s, k]
                    output[b, s, :] += prob * results[i]
        else:
            # Single GPU processing
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