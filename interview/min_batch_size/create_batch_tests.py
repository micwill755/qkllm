# Large test case for min_batch_size.py
import random

def generate_large_test_case():
    # Generate 10,000 data samples with varying sizes
    random.seed(42)  # For reproducible results
    
    data_samples = []
    
    # Mix of small, medium, and large data samples
    for i in range(10000):
        if i < 3000:  # Small samples (1-100)
            data_samples.append(random.randint(1, 100))
        elif i < 7000:  # Medium samples (100-10000)
            data_samples.append(random.randint(100, 10000))
        else:  # Large samples (10000-1000000)
            data_samples.append(random.randint(10000, 1000000))
    
    max_batches = 50000  # Allow up to 50,000 batches
    
    print(f"Test case: {len(data_samples)} samples, max_batches = {max_batches}")
    print(f"Sample range: {min(data_samples)} to {max(data_samples)}")
    
    return data_samples, max_batches

# Generate and save test case
data_samples, max_batches = generate_large_test_case()

# Save to file for testing
with open('./interview/large_test_input.txt', 'w') as f:
    f.write(f"{len(data_samples)}\n")
    for sample in data_samples:
        f.write(f"{sample}\n")
    f.write(f"{max_batches}\n")

print("Large test case saved to 'large_test_input.txt'")