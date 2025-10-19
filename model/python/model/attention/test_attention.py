import numpy as np
from model.python.model.attention.attention import MultiHeadAttention

def test_multihead_attention():
    # Test parameters
    np.random.seed(42)
    d_in = 64
    d_out = 64
    context_length = 8
    num_heads = 4
    batch_size = 2
    seq_len = 4
    
    # Create attention layer
    attention = MultiHeadAttention(d_in, d_out, context_length, 0.1, num_heads)
    
    # Test input
    x = np.random.randn(batch_size, seq_len, d_in)
    
    # Forward pass
    output = attention.forward(x)
    
    # Assertions
    assert output.shape == (1, batch_size, seq_len, d_out), f"Expected shape (1, {batch_size}, {seq_len}, {d_out}), got {output.shape}"
    assert not np.isnan(output).any(), "Output contains NaN values"
    assert not np.isinf(output).any(), "Output contains infinite values"
    
    print("✓ Shape test passed")
    print("✓ NaN/Inf test passed")
    print("✓ All tests passed!")

if __name__ == "__main__":
    test_multihead_attention()