import math

class RMSNorm():
    def __init__(self, emb_dim, eps=1e-5):
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = [1.0 for _ in range(emb_dim)]

    def forward(self, x):
        # x is a flattened 1D array: [token0_emb, token1_emb, ...]
        # Need to normalize each token's embedding separately
        seq_len = len(x) // self.emb_dim
        result = [0 for _ in range(len(x))]
        
        for token_idx in range(seq_len):
            start_idx = token_idx * self.emb_dim
            end_idx = start_idx + self.emb_dim
            
            # Calculate RMS for this token's embedding
            sum_squares = 0.0
            for i in range(start_idx, end_idx):
                sum_squares += x[i] * x[i]
            
            rms = math.sqrt(sum_squares / self.emb_dim + self.eps)
            
            # Normalize and apply weight
            for i in range(self.emb_dim):
                result[start_idx + i] = (x[start_idx + i] / rms) * self.weight[i]
        
        return result

class LayerNorm():
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        self.eps = 1e-5
        self.scale = [1 for _ in range(emb_dim)]
        self.shift = [0 for _ in range(emb_dim)]

    def forward(self, x):
        # x is flattened: [token0_emb, token1_emb, ...]
        seq_len = len(x) // self.emb_dim
        result = [0 for _ in range(len(x))]
        
        for token_idx in range(seq_len):
            start_idx = token_idx * self.emb_dim
            end_idx = start_idx + self.emb_dim
            
            # Calculate mean for this token's embedding
            mean = sum(x[start_idx:end_idx]) / self.emb_dim
            
            # Calculate variance
            var = sum((x[i] - mean) ** 2 for i in range(start_idx, end_idx)) / self.emb_dim
            
            # Normalize and apply scale/shift
            for i in range(self.emb_dim):
                norm_val = (x[start_idx + i] - mean) / math.sqrt(var + self.eps)
                result[start_idx + i] = self.scale[i] * norm_val + self.shift[i]
        
        return result

def test_rms_norm():
    emb_dim = 4
    rms_norm = RMSNorm(emb_dim)
    
    x = [1.0, 2.0, 3.0, 4.0,  # token 1
         5.0, 6.0, 7.0, 8.0]  # token 2
    
    result = rms_norm.forward(x)
    
    for token_idx in range(2):
        start = token_idx * emb_dim
        end = start + emb_dim
        token_vals = result[start:end]
        
        mean = sum(token_vals) / emb_dim
        rms = math.sqrt(sum(v**2 for v in token_vals) / emb_dim)
        
        print(f"Token {token_idx}: mean={mean:.6f}, rms={rms:.6f}")
        print(f"  Values: {[f'{v:.4f}' for v in token_vals]}")

def test_layer_norm():
    emb_dim = 4
    norm = LayerNorm(emb_dim)
    
    # 2 tokens, each with 4-dimensional embeddings
    x = [1.0, 2.0, 3.0, 4.0,  # token 1
         5.0, 6.0, 7.0, 8.0]  # token 2
    
    result = norm.forward(x)
    
    # Verify each token is normalized to mean≈0, std≈1
    for token_idx in range(2):
        start = token_idx * emb_dim
        end = start + emb_dim
        token_vals = result[start:end]
        
        mean = sum(token_vals) / emb_dim
        var = sum((v - mean) ** 2 for v in token_vals) / emb_dim
        std = math.sqrt(var)
        
        print(f"Token {token_idx}: mean={mean:.6f}, std={std:.6f}")
        print(f"  Values: {[f'{v:.4f}' for v in token_vals]}")

if __name__ == "__main__":
    print("=== LayerNorm ===")
    test_layer_norm()
    print("\n=== RMSNorm ===")
    test_rms_norm()
