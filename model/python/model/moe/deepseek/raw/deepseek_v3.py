import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from ROPE import ROPE, test_rope
from model.linear import Linear, Linear1d
from model.lib.matrix_helper import mat_mul_1d, apply_cause_mask_1d
from model.norm import RMSNorm
from model.python.model.feedforward import Expert
from model.embedding import Embedding1d
from model.moe.expert_tracker import ExpertTracker

import tiktoken

class MoE():
    '''
    emb_dim: Input/output dimension
    num_experts: Total number of experts (e.g., 8, 16)
    top_k: Number of experts to activate (usually 2)
    expert_dim: Hidden dimension in each expert FFN
    '''
    def __init__(self, emb_dim, num_experts, top_k, expert_dim, load_balance_weight=0.01):
        self.router = Linear1d(emb_dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = [Expert(emb_dim, expert_dim) for _ in range(num_experts)]
        self.tracker = ExpertTracker(num_experts)
        self.load_balance_weight = load_balance_weight

    def forward(self, x, batch, seq_len, emb_dim):
        '''
            For Each Token:
            Extract token embedding: Get emb_dim elements for each token
            Apply router: Pass token embedding through self.router.forward()
            Get expert scores: Router outputs num_experts scores for this token
        '''
        # step 1 router computation - 1d array we can simply pass the entire array in
        expert_scores = self.router.forward(x)

        # step 2 - softmax scores to probabilities
        for row in range(seq_len):
            ma = float('-inf')
            ex_sum = 0

            # get max of row
            for col in range(self.num_experts):
                if expert_scores[row * self.num_experts + col] > ma:
                    ma = expert_scores[row * self.num_experts + col]

            for col in range(self.num_experts):
                exp = math.exp(expert_scores[row * self.num_experts + col] - ma)
                ex_sum += exp
                expert_scores[row * self.num_experts + col] = exp

            for col in range(self.num_experts):
                expert_scores[row * self.num_experts + col] /= ex_sum

        all_top_k_experts = []

        # step 3 - top k take the top 2 highest probabiliti experts for every token
        for row in range(seq_len):
            # approach 1 (brute force)
            # find top k experts for this token
            '''for _ in range(self.top_k):
                max_prob = float('-inf')
                max_idx = -1

                # find highest remaining probability
                for col in range(self.num_experts):
                    prob =  expert_scores[row * self.num_experts + col]
                    if prob > max_prob:
                        max_idx = col
                        max_prob = prob
                
                # Store this expert and zero it out
                top_k_experts.append((max_idx, max_prob))
                expert_scores[row * self.num_experts + col] = 0
                
            # Store this expert and zero it out
            top_k_experts.append((max_idx, max_prob))
            expert_scores[row * self.num_experts + col] = 0
            '''

            # approach 2 (sorting) - TODO: for now this sorting makes it easier and more efficient, need to come back to improve
            expert_probs = []

            for col in range(self.num_experts):
                prob = expert_scores[row * self.num_experts + col]
                expert_probs.append((prob, col))
            
            expert_probs.sort(reverse=True)

            selected_experts = []
            for expert_prob in expert_probs[:self.top_k]:
                all_top_k_experts.append(expert_prob)
                selected_experts.append(expert_prob[1])  # expert index
            
            # Track expert usage
            self.tracker.update(selected_experts)

        # step 4 Expert routing and weighted combination

        out = [0.0 for _ in range(len(x))]

        for row in range(seq_len):
            # get this token's embedding
            token_start = row * emb_dim
            token_embedding = x[token_start:token_start + emb_dim]

            # weighted sum is how you combine multiple expert outputs into a single final output for each token.

            '''
            example

            Token input: [1, 2, 3, 4]
            Selected experts: Expert 2 (prob=0.6), Expert 5 (prob=0.4)

            Expert outputs:

            Expert 2 processes [1,2,3,4] → outputs [10, 20, 30, 40]

            Expert 5 processes [1,2,3,4] → outputs [5, 15, 25, 35]

            Weighted sum calculation:

            weighted_sum[0] = 0.6 * 10 + 0.4 * 5  = 6 + 2   = 8
            weighted_sum[1] = 0.6 * 20 + 0.4 * 15 = 12 + 6  = 18
            weighted_sum[2] = 0.6 * 30 + 0.4 * 25 = 18 + 10 = 28
            weighted_sum[3] = 0.6 * 40 + 0.4 * 35 = 24 + 14 = 38

            Copy
            Final token output: [8, 18, 28, 38]

            Why weighted? The router learned that Expert 2 is more relevant (60%) than Expert 5 (40%) 
            for this specific token, so Expert 2's output has more influence in the final result. 
            This creates a smooth blend rather than a hard choice.
            
            '''

            weighted_sum = [0.0 for _ in range(emb_dim)]
            # all_top_k_experts is flattened, so we have to get just the self.top_k experts for this token
            top_k_token_start = row * self.top_k

            for top_k_idx in range(top_k_token_start, top_k_token_start + self.top_k):
                prob, expert_idx = all_top_k_experts[top_k_idx]
                expert_output = self.experts[expert_idx].forward(token_embedding)
                # weight by routing probability and add to sum
                for i in range(emb_dim):
                    weighted_sum[i] += prob * expert_output[i]

            # Store result back to output
            for i in range(emb_dim):
                out[token_start + i] = weighted_sum[i]

        return out
    
    def get_expert_stats(self):
        """Get current expert utilization statistics"""
        utilization = self.tracker.get_utilization()
        unused = self.tracker.get_unused_experts()
        return {
            'utilization': utilization,
            'unused_experts': unused,
            'total_tokens': self.tracker.total_tokens
        }
    
    def apply_load_balancing_loss(self, expert_scores, seq_len):
        """Apply load balancing loss to encourage expert diversity"""
        # Calculate expert usage frequency
        expert_usage = [0.0] * self.num_experts
        
        for row in range(seq_len):
            for col in range(self.num_experts):
                expert_usage[col] += expert_scores[row * self.num_experts + col]
        
        # Normalize by sequence length
        for i in range(self.num_experts):
            expert_usage[i] /= seq_len
        
        # Calculate load balancing loss (encourages uniform distribution)
        mean_usage = sum(expert_usage) / self.num_experts
        load_loss = sum((usage - mean_usage) ** 2 for usage in expert_usage)
        
        return load_loss * self.load_balance_weight
    
    def reinitialize_unused_experts(self, threshold=0.001):
        """Reinitialize experts that are rarely used"""
        unused = self.tracker.get_unused_experts(threshold)
        for expert_idx in unused:
            # Reinitialize the expert with new random weights
            self.experts[expert_idx] = Expert(self.experts[expert_idx].linear1.input_dim, 
                                            self.experts[expert_idx].linear2.input_dim)
        return unused


class MHLA:
    # emb_dim = also known as d_model, is the size of the token embeddings eg. 768
    def __init__(self, emb_dim, num_heads, max_seq_length):
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.rope = ROPE(emb_dim=emb_dim, max_seq_length=max_seq_length)

        self.query_W = Linear1d(emb_dim, emb_dim)
        self.key_W = Linear1d(emb_dim, emb_dim)
        self.value_W = Linear1d(emb_dim, emb_dim)
        
        self.output_proj = Linear1d(emb_dim, emb_dim)

    def transpose(self, m, batch, seq_len, num_heads, head_dim):
        out = [0 for _ in m]

        for row in range(seq_len):
            for col in range(head_dim):
                input_idx = row * head_dim + col
                output_idx = col * seq_len + row
                out[output_idx] = m[input_idx]

        return out
    
    # temp until me move tranpose to matrix helper
    def test_transpose(self):
        print("Testing transpose implementation...")
        
        # Test simple 2x2 matrix for one head
        # Original: [1,2]  -> Transposed: [1,3]
        #           [3,4]                  [2,4]
        # 1D format: [1,2,3,4] -> [1,3,2,4]

        # Test data: 1 batch, 2 tokens, 1 head, 2 dims per head
        test_matrix = [1.0, 2.0, 3.0, 4.0]  # [seq=2, head_dim=2]
        
        result = self.transpose(test_matrix, batch=1, seq_len=2, num_heads=1, head_dim=2)
        
        print(f"Input:  {test_matrix}")
        print(f"Output: {result}")
        print(f"Expected: [1.0, 3.0, 2.0, 4.0]")
        
        # Verify the transpose worked correctly
        expected = [1.0, 3.0, 2.0, 4.0]
        if result == expected:
            print("Transpose test passed!")
        else:
            print("Transpose test failed!")
        
        return result == expected

    def forward(self, x, batch, seq_len, emb_dim):        
        # step 1 perform mx + b (linear transformation) on input embeddings, but most modern attention mechanisms dont use a bias term (b)
        # so its really just mx
        query = self.query_W.forward(x)
        key = self.key_W.forward(x)
        value = self.value_W.forward(x)

        # step 2 split each matrix into seperate head dimensions [b, tokens, emd_dim] -> [b, tokens, heads, head_dim]
        # since we are using 1d arrays we dont need to reshape

        # step 3 apply rope rotation to query and key matrix
        query_r = self.rope.forward(query, batch, seq_len, emb_dim)
        key_r = self.rope.forward(key, batch, seq_len, emb_dim)
            
        # step 4 scale dot product attention Q @ transpose (keys) / sqrt(emb_dim)
        keys_T = self.transpose(key_r, batch, seq_len, self.num_heads, self.head_dim)
        attn_scores = mat_mul_1d(query_r, keys_T, seq_len, self.head_dim, self.head_dim, seq_len)

        # because we're using Python lists (1D array format), not NumPy arrays, you can't do vectorized operations like division.
        # step 5 scale the attention scores
        scale = 1 / math.sqrt(self.head_dim)
        attn_weights = [scale * s for s in attn_scores]

        # step 6 mask future tokens
        apply_cause_mask_1d(attn_weights, seq_len)

        # step 7 apply soft max
        for row in range(seq_len):
            # Find max of non-masked values
            ma = -float('inf')
            for col in range(seq_len):
                if attn_weights[row * seq_len + col] > ma:
                    ma = attn_weights[row * seq_len + col]
            
            # Apply softmax: exp(x - max) for non-masked, 0 for masked
            sum_ex = 0.0
            for col in range(seq_len):
                exp_val = math.exp(attn_weights[row * seq_len + col] - ma)
                attn_weights[row * seq_len + col] = exp_val
                sum_ex += exp_val
            
            # Normalize to sum to 1 (only non-masked values)
            if sum_ex > 0:
                for col in range(seq_len):
                    if attn_weights[row * seq_len + col] > 0:  # Only normalize non-zero values
                        attn_weights[row * seq_len + col] /= sum_ex

        # step 8 apply attention to valyes
        attn_output = mat_mul_1d(attn_weights, value, seq_len, seq_len, seq_len, emb_dim)

        # step 9 no need to combine heads because we are using a 1d matrix

        # step 10 perform mx + b (linear transformation) with attn output and output_proj matrix
        return self.output_proj.forward(attn_output)

'''
The Block should follow the pre-norm architecture that DeepSeek uses:

Input (x)
    ↓
RMSNorm → MHLA → Add with residual (x + MHLA_output)
    ↓
RMSNorm → MoE/FFN → Add with residual (x + MoE_output)
    ↓
Output

'''
class Block():
    def __init__(self, batch, emb_dim, num_heads, seq_len, num_experts, top_k, expert_dim):
        self.batch = batch
        self.emb_dim = emb_dim
        self.seq_len = seq_len

        self.rmsNorm1 = RMSNorm(emb_dim)
        self.MHLA = MHLA(emb_dim, num_heads, seq_len)
        self.rmsNorm2 = RMSNorm(emb_dim)
        self.moE = MoE(emb_dim, num_experts, top_k, expert_dim)

    def forward(self, x):
        # FIRST SUB-LAYER: Multi-Head Latent Attention with Shortcut Connection
        # Pre-norm: Normalize input before processing (DeepSeek uses pre-norm architecture)
        normalized_x = self.rmsNorm1.forward(x)

        # Apply attention mechanism to normalized input
        attn_output = self.MHLA.forward(normalized_x, self.batch, self.seq_len, self.emb_dim)

        # SHORTCUT CONNECTION #1: Add original input to attention output
        for i in range(len(x)):
            x[i] += attn_output[i]  # Residual/Skip connection around attention

        # SECOND SUB-LAYER: Mixture of Experts with Shortcut Connection  
        # Pre-norm: Normalize the output from first sub-layer before MoE processing
        normalized_x = self.rmsNorm2.forward(x)

        # Apply MoE layer to normalized input (sparse expert routing)
        moe_output = self.moE.forward(normalized_x, self.batch, self.seq_len, self.emb_dim)

        # SHORTCUT CONNECTION #2: Add input to MoE output
        for i in range(len(x)):
            x[i] += moe_output[i]  # Residual/Skip connection around MoE

        # Return final output with both shortcut connections applied
        # Each block learns incremental transformations rather than complete mappings
        return x

class DeepSeekV3Model:
    def __init__(self, cfg):
        self.tok_emb = Embedding1d(cfg["vocab_size"], cfg["emb_dim"])
        self.blocks = [Block(cfg["batch"], cfg["emb_dim"], cfg["num_heads"], 
                    cfg["seq_len"], cfg["num_experts"], cfg["top_k"], 
                    cfg["expert_dim"]) for _ in range(cfg["n_layers"])]
        self.finalRMSNorm = RMSNorm(cfg["emb_dim"])
        self.out_head = Linear1d(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, x, batch, seq_len, emb_dim):
        # Convert token IDs to flattened embeddings
        x = [self.tok_emb.weight[token_id * emb_dim + i] 
             for token_id in x 
             for i in range(emb_dim)]
        
        for block in self.blocks:
            x = block.forward(x)
            
        x = self.finalRMSNorm.forward(x)
        logits = self.out_head.forward(x)
        return logits

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

#model = DeepSeekV3Model(cfg)

# Unit Tests
def test_deepseek_model():
    print("Testing DeepSeek V3 Model...")
    
    # Small config for fast testing
    test_cfg = {
        "vocab_size": 50257,  # GPT-2 vocab size (needed for real tokens)
        "emb_dim": 64,        # Small embedding
        "n_layers": 2,        # Only 2 layers
        "batch": 1,
        "num_heads": 4,       # 4 heads (64/4 = 16 head_dim)
        "seq_len": 8,         # Short sequence
        "num_experts": 4,     # 4 experts
        "top_k": 2,           # Top-2 routing
        "expert_dim": 128     # Small expert hidden dim
    }
    
    # Create model
    test_model = DeepSeekV3Model(test_cfg)
    
    # Test input: use small token IDs first to debug
    '''test_input = [1, 2, 3, 4, 5, 6, 7, 8]  # Simple small token IDs
    
    print(f"Test tokens: {test_input}")
    print(f"Token count: {len(test_input)}")'''
    
    # Uncomment below for real tokenizer test once basic test works
    test_text = "Hello world, this is a test."
    test_input = tokenizer.encode(test_text)[:test_cfg["seq_len"]]
    
    # Forward pass
    try:
        logits = test_model.forward(
            test_input, 
            test_cfg["batch"], 
            test_cfg["seq_len"], 
            test_cfg["emb_dim"]
        )
        
        # Check output shape
        expected_size = test_cfg["seq_len"] * test_cfg["vocab_size"]  # 8 * 50257 = 402056
        
        if len(logits) == expected_size:
            print("Model forward pass successful!")
            print(f"Output shape correct: {len(logits)} (expected {expected_size})")
            print(f"Sample logits: {logits[:5]}")  # Show first 5 values
            return True
        else:
            print(f"Wrong output shape: {len(logits)} (expected {expected_size})")
            return False
            
    except Exception as e:
        print(f"Model forward pass failed: {e}")
        return False

def test_model_components():
    print("\nTesting Individual Components...")
    
    # Test MHLA transpose
    mhla = MHLA(emb_dim=64, num_heads=4, max_seq_length=8)
    transpose_result = mhla.test_transpose()
    
    if transpose_result:
        print("MHLA transpose test passed!")
    else:
        print("MHLA transpose test failed!")
    
    return transpose_result

# Run tests
if __name__ == "__main__":
    print("Running DeepSeek V3 Tests...\n")
    
    # Test individual components
    component_test = test_model_components()
    
    # Test full model
    model_test = test_deepseek_model()
    
    # Summary
    print(f"\nTest Results:")
    print(f"Components: {'PASS' if component_test else 'FAIL'}")
    print(f"Full Model: {'PASS' if model_test else 'FAIL'}")
    
    if component_test and model_test:
        print("\nAll tests passed! Your DeepSeek V3 model is working correctly!")
    else:
        print("\nSome tests failed. Check the error messages above.")