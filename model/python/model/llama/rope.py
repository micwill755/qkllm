
import math
import tiktoken 
import random
import copy

# using torch - TODO: come back to review implementation
'''
class ROPE:
    def __init__(self, emb_dim, max_seq_len=2048, base=10000):
        # precompute frequencies and rotation matrices
        self.pairs = emb_dim // 2
        
        # Step 1: Create frequency for each dimension pair
        # θᵢ = base^(-2i/d) where i goes from 0 to d/2
        inv_freq = 1.0 / (base ** (torch.arange(0, emb_dim, 2).float() / emb_dim))
        self.register_buffer('inv_freq', inv_freq)

        pass

    def _get_cos_sin(self, seq_len):
        # Step 2: Create position indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        
        # Step 3: Outer product gives us all position-frequency combinations
        # Shape: (seq_len, emb_dim//2)
        freqs = torch.outer(t, self.inv_freq)
        
        # Step 4: Get cos and sin for rotation
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    def __call__(self, x):
        self.forward(x)

    def forward(self, x):
        # x shape: (batch, seq_len, heads, head_dim)
        seq_len = x.shape[1]
        cos, sin = self._get_cos_sin(seq_len)

        # Step 5: Split into pairs for rotation
        # Treat (x[..., 0], x[..., 1]) as real and imaginary parts
        x1 = x[..., ::2]   # Even indices (real part)
        x2 = x[..., 1::2]  # Odd indices (imaginary part)

        # Step 6: Apply rotation formula
        # Real part: x1*cos - x2*sin
        # Imag part: x1*sin + x2*cos
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Step 7: Interleave back
        out = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return out.flatten(-2)
'''

class RoPE:
    def __init__(self, emb_dim, max_seq_len=2048, base=10000):
        self.emb_dim = emb_dim
        self.pairs = self.emb_dim // 2

        # Precompute frequencies for each dimension pair
        self.inv_freq = []
        for i in range(0, emb_dim, 2):
            freq = 1.0 / (base ** (i / emb_dim))
            self.inv_freq.append(freq)
    
    '''
    When computing attention between two positions:

    - Fast frequencies help the model know "this word is right next to that word"
    - Slow frequencies help the model know "this sentence is near the beginning, that one is near the end"

    The slowest frequency gives the model a sense of global position within the entire sequence - whether tokens are in the beginning, middle, or end of a long document.

    '''
    
    def get_cos_and_sin(self, seq_len):
        # Create cos/sin tables for all positions and frequencies
        cos_table = []
        sin_table = []

        for pos in range(seq_len):
            cos_row = []
            sin_row = []
            for freq in self.inv_freq:
                angle = pos * freq
                cos_row.append(math.cos(angle))
                sin_row.append(math.sin(angle))
            cos_table.append(cos_row)
            sin_table.append(sin_row)

        return cos_table, sin_table

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x shape: (batch, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = x.shape
        cos_table, sin_table = self.get_cos_and_sin(seq_len)

        for b in range(batch_size):
            for h in range(num_heads):
                for pos in range(seq_len):
                    for pair_idx in range(self.pairs):
                        dim1_idx = pair_idx * 2
                        dim2_idx = pair_idx * 2 + 1
                        
                        x1 = x.tensor[b][h][pos][dim1_idx]
                        x2 = x.tensor[b][h][pos][dim2_idx]
                        
                        cos_val = cos_table[pos][pair_idx]
                        sin_val = sin_table[pos][pair_idx]
                        
                        new_x1 = x1 * cos_val - x2 * sin_val
                        new_x2 = x1 * sin_val + x2 * cos_val
                        
                        x.tensor[b][h][pos][dim1_idx] = new_x1
                        x.tensor[b][h][pos][dim2_idx] = new_x2
        
        return x

if __name__ == "__main__":
    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.n_vocab  # ~50k tokens
    emb_dim = 8
    
    # Initialize embedding layer
    embedding = Embedding(vocab_size, emb_dim)
    
    # Tokenize and embed
    prompt = "Hello world, this is a test"
    tokens = encoder.encode(prompt)
    embeddings = embedding.forward(tokens)
    
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {tokens}")
    print(f"Embedding shape: seq_len={len(embeddings)}, emb_dim={len(embeddings[0])}")
    
    # Show original embeddings
    print(f"\nOriginal embeddings:")
    for i, emb in enumerate(embeddings):
        print(f"Token {tokens[i]}: {[round(x, 3) for x in emb[:4]]}...")
    
    # After getting embeddings, make a copy before RoPE
    original_embeddings = copy.deepcopy(embeddings)

    # Apply RoPE - final embedding values
    batch_embeddings = [embeddings]  # add batch dimension
    rope = RoPE(emb_dim=emb_dim)
    rotated = rope(batch_embeddings)
    
    print(f"\nAfter RoPE:")
    for i, emb in enumerate(rotated[0]):
        print(f"Position {i}: {[round(x, 3) for x in emb[:4]]}...")
    
    # Frequency hierarchy working: Fast frequencies create bigger rotations, slow frequencies create smaller ones
    # Position-dependent: Each position gets different rotation angles, creating unique positional signatures
    # Relative encoding: When computing attention between positions, the dot product will naturally encode their relative distance through these rotations

    # below are the changes in the embedding values
    '''
        The Relationship:
        Original + Change = Final

        eg:
        Original: [-0.05, -0.011, -0.001, 0.047]
        Change: [0.032, -0.037, -0.005, -0.0]
        Final: [-0.018, -0.048, -0.006, 0.047]
        You can verify: -0.05 + 0.032 = -0.018 ✓
    '''

    print(f"Rotation effect (first 4 dims):")
    for i in range(len(original_embeddings)):
        orig = original_embeddings[i][:4]
        rot = rotated[0][i][:4]
        diff = [rot[j] - orig[j] for j in range(4)]
        print(f"Position {i} diff: {[round(x, 3) for x in diff]}")
