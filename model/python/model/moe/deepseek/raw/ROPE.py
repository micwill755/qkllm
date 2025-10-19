import math

class ROPE:
    '''
    emd_dim - token embedding dimension - must be even for rotation pairs
    max_seq_length - max sequence length to pre compute rotations
    base 
    '''
    def __init__(self, emb_dim, max_seq_length=2048, base=10000):
        self.pairs_per_token = emb_dim // 2
        self.frequencies = [(1 / base**(2 * i/emb_dim)) for i in range(emb_dim // 2)]
        self.positions = [i for i in range(max_seq_length)]
        self.angles = [p * f for p in self.positions for f in self.frequencies]

    # x - 1d array, b = batch, s = sequence, d = emb_dim
    def forward(self, x, batch, seq_len, emb_dim):
        #angles = self.angles[0:seq_len]
        # the next step is to reshape x but we dont need to because we are using 1d arrays
        # future work x = reshape_1d(x, (batch, seq_len, emb_dim), (batch, seq_len, emb_dim // 2, 2))
        #x = reshape_1d_dim_pairs(x, batch, seq_len, emb_dim)
        for i in range(0, len(x), 2):
            pair_idx = i // 2
            X = x[i]
            Y = x[i + 1]
            angle = self.angles[pair_idx]
            # TODO: NEED TO UNDER STAND THIS IMPLEMENTATION
            new_x = X * math.cos(angle) - Y * math.sin(angle)
            new_y = X * math.sin(angle) + Y * math.cos(angle)
            x[i] = new_x
            x[i + 1] = new_y
        
        return x

# Simple unit test for ROPE
def test_rope():
    print("Testing ROPE implementation...")
    
    # Test parameters
    batch = 1
    seq_len = 2
    emb_dim = 4
    
    # Create test data: simple 1D array
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # batch=1, seq=2, emb=4
    print(f"Input: {x}")
    
    # Initialize ROPE
    rope = ROPE(emb_dim=emb_dim, max_seq_length=10, base=10000)
    print(f"Frequencies: {rope.frequencies}")
    print(f"First few angles: {rope.angles[:8]}")
    
    # Apply ROPE
    rope.forward(x, batch, seq_len, emb_dim)
    print(f"Output: {x}")
    
    # Basic sanity checks
    assert len(x) == 8, "Output length should match input"
    assert x != [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "Values should have changed"
    
    print("✅ ROPE test passed!")

if __name__ == "__main__":
    test_rope()