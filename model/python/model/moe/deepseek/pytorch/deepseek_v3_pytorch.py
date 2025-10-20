import math
import sys
import os

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import tiktoken

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

class RoPE(nn.Module):
    def __init__(self, emb_dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.emb_dim = emb_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, emb_dim, 2).float() / emb_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        # x shape: (batch, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Create position indices and compute frequencies
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim//2)
        
        cos = torch.cos(freqs)  # (seq_len, head_dim//2)
        sin = torch.sin(freqs)
        
        # Reshape x into pairs: (batch, num_heads, seq_len, head_dim//2, 2)
        x_pairs = x.view(batch_size, num_heads, seq_len, head_dim // 2, 2)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
        
        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Stack and reshape back
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return rotated.view(batch_size, num_heads, seq_len, head_dim)

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
        self.router = nn.Linear(emb_dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        '''
        Parameter Registration: nn.ModuleList automatically registers all expert parameters with PyTorch, so they'll be included in:
        model.parameters() (for optimization)
        model.state_dict() (for saving/loading)
        Device transfers (model.to(device))
        '''
        self.experts = nn.ModuleList([Expert(emb_dim, expert_dim) for _ in range(num_experts)])

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

        for b in range(batch_size):
            for s in range(seq_len):
                token_embedding = x[b, s, :]  # (emb_dim,)
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

        self.rope = RoPE(emb_dim=self.head_dim, max_seq_len=max_seq_length)

        self.latent_tokens = nn.Linear(max_seq_length * emb_dim, latent_dim * emb_dim)
        self.query_W = nn.Linear(emb_dim, emb_dim)
        self.key_W = nn.Linear(emb_dim, emb_dim)
        self.value_W = nn.Linear(emb_dim, emb_dim)
        self.output_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        
        x_flat = x.view(batch_size, -1)  # (batch, seq_len * emb_dim)
        latent_compressed = self.latent_tokens(x_flat)  # (batch, latent_dim * emb_dim)
        latent = latent_compressed.view(batch_size, self.latent_dim, emb_dim)

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
        
        # step 3: Apply RoPE AFTER reshaping (standard practice)
        query_r = self.rope(query)     # (batch, num_heads, seq_len, head_dim)
        key_r = key                    # No RoPE for latent tokens
            
        # step 4: attention computation (now per head)
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

# inference test -----

def text_to_tokens(text, tokenizer):
    encoded_tokens = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded_tokens).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, input_ids, max_new_tokens, cfg, pad_token):
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # add padding to sequence if input is less than seq len
            padded_sequences = []
            for i in input_ids:
                if len(i) < cfg["seq_len"]:
                    pad_len = cfg["seq_len"] - len(i)
                    padding = torch.full((pad_len,), pad_token, dtype=i.dtype, device=i.device)
                    padded = torch.cat([i, padding])
                else:
                    padded = i[:cfg["seq_len"]]
                padded_sequences.append(padded)
            
            # Stack back into batch tensor
            padded_input = torch.stack(padded_sequences)

            logits = model(padded_input)
            
            # Get next token probabilities (last position)
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we hit sequence limit
            if input_ids.size(1) >= cfg["seq_len"]:
                break

    return input_ids

# inference -----

# Run tests
if __name__ == "__main__":
    print("Running DeepSeek V3 Tests...\n")
    
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
        "expert_dim": 128,     # Small expert hidden dim
        "context_length": 512
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    model = DeepSeekV3Model(test_cfg)

    # un trained output
    '''prompt = "Hello world, my"
    enc_tensor = text_to_tokens(prompt, tokenizer)
    print("Input shape", enc_tensor.shape)
    out = generate_text_simple(
        model, 
        enc_tensor, 
        5,
        test_cfg,
        50256)

    output = token_ids_to_text(out, tokenizer)
    print(output)'''

    inputs = torch.tensor([[40, 716, 281, 27140, 44, 508, 7832, 284]]) # "I am an LLM who likes to"
    print(inputs.shape)
    print(token_ids_to_text(inputs, tokenizer))
    targets = torch.tensor([[321, 281, 27140, 44, 508, 7832, 284, 2193]]) # "am an LLM who likes to learn"
    print(token_ids_to_text(targets, tokenizer))

    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
    print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    
    print("Token IDs:\n", token_ids)
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    
    target_probas_1 = probas[0, torch.arange(8), targets[0]]
    print("Text 1:", target_probas_1)

    log_probas = torch.log(target_probas_1)
    print("Log probabilities", log_probas)

    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)

    #model_test = test_deepseek_model(model)
    '''model = DeepSeekV3Model(test_cfg)
    
    if model_test:
        print("\All tests passed! Your DeepSeek V3 model is working!")
        dummy_input = torch.randint(0, test_cfg["vocab_size"], (1, test_cfg["seq_len"]))
        onnx_path = "deepseek_v3.onnx"
        if not os.path.exists(onnx_path):
            torch.onnx.export(model, dummy_input, onnx_path,
                input_names=['input_ids'], output_names=['logits'])
    else:
        print("\n❌ Some tests failed. Check the error messages above.")'''

    