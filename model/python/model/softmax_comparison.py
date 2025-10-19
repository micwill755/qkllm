import numpy as np

def standard_softmax(scores):
    """Standard softmax - needs full row at once"""
    max_val = np.max(scores)
    exp_scores = np.exp(scores - max_val)
    return exp_scores / np.sum(exp_scores)

def online_softmax(score_blocks):
    """Online softmax - processes blocks incrementally"""
    row_max = -np.inf
    row_sum = 0.0
    result = []
    
    # Store intermediate values for final normalization
    all_exp_scores = []
    all_maxes = []
    
    for block in score_blocks:
        block_max = np.max(block)
        new_max = max(row_max, block_max)
        
        # Current block exponentials
        exp_scores = np.exp(block - new_max)
        
        # Rescale previous sum if max changed
        exp_prev = np.exp(row_max - new_max) if row_max != -np.inf else 0
        row_sum = row_sum * exp_prev + np.sum(exp_scores)
        
        # Store for final result
        all_exp_scores.append(exp_scores)
        all_maxes.append(new_max)
        row_max = new_max
    
    # Final normalization
    final_result = []
    for exp_scores in all_exp_scores:
        final_result.extend(exp_scores / row_sum)
    
    return np.array(final_result)

# Example usage
if __name__ == "__main__":
    # Test scores
    scores = np.array([2.0, 1.0, 3.0, 0.5])
    
    # Standard softmax
    standard_result = standard_softmax(scores)
    
    # Online softmax (split into blocks)
    blocks = [scores[:2], scores[2:]]  # [2.0, 1.0] and [3.0, 0.5]
    online_result = online_softmax(blocks)
    
    print("Original scores:", scores)
    print("Standard softmax:", standard_result)
    print("Online softmax:  ", online_result)
    print("Results match:   ", np.allclose(standard_result, online_result))
    
    # Show step-by-step online process
    print("\n--- Online Softmax Steps ---")
    row_max = -np.inf
    row_sum = 0.0
    
    for i, block in enumerate(blocks):
        print(f"\nBlock {i+1}: {block}")
        block_max = np.max(block)
        new_max = max(row_max, block_max)
        
        exp_scores = np.exp(block - new_max)
        exp_prev = np.exp(row_max - new_max) if row_max != -np.inf else 0
        
        print(f"  block_max: {block_max}")
        print(f"  new_max: {new_max}")
        print(f"  exp_scores: {exp_scores}")
        print(f"  exp_prev: {exp_prev}")
        print(f"  old_sum: {row_sum}")
        
        row_sum = row_sum * exp_prev + np.sum(exp_scores)
        print(f"  new_sum: {row_sum}")
        row_max = new_max