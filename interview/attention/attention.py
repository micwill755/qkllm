import math
import qkmx.mtrx
from qkmx.tensor import Tensor
from qkmx.mtrx import softmax, dot_product, mat_mul, mask, reshape
from qkmx.linear import Linear

#inputs = torch.rand(6, 3)
# we are hard coding to see the values 
inputs = Tensor([[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
    )

# 1. compute context vector for a single query token which is journey

'''att_scores = Tensor((inputs.shape[0], inputs.shape[0]))
for i in range(len(inputs)):
    for j in range(len(inputs)):
        att_scores[i][j] = round(dot_product(inputs[i], inputs[j]), 4)

print(att_scores)
att_weights = softmax(att_scores)

query = inputs[1]
context = torch.torch.zeros(query.shape)

for i in range(context.shape[0]):
    weighted_sum = 0
    for j in range(len(att_weights)):
        weighted_sum += att_weights[1][j] * inputs[j][i]
    context[i] = round(weighted_sum, 4)

print(context)'''

# 2. compute attention weights for all tokens using matrix multiplication
'''context = torch.torch.zeros(inputs.shape)
att_scores = mat_mul(inputs, inputs.T)
print(inputs.T)
att_weights = softmax(att_scores)
context = mat_mul(att_weights, inputs)
print(context)'''

# implement self attention with trainable weights
class SelfAttention:
    def __init__(self, d_in, d_out):
        self.q_W = Linear(d_in, d_out)
        self.k_W = Linear(d_in, d_out)
        self.v_W = Linear(d_in, d_out)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        att_scores = mat_mul(query, key.T)
        att_weights = softmax(att_scores)
        context = mat_mul(att_weights, value)
        return context
    
'''attention = SelfAttention(inputs.shape[1], inputs.shape[1])
out = attention(inputs)
print(out)'''

class SingleHeadCasualAttention:
    def __init__(self, d_in, d_out):
        self.q_W = Linear(d_in, d_out)
        self.k_W = Linear(d_in, d_out)
        self.v_W = Linear(d_in, d_out)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        att_scores = mat_mul(query, key.T)
        att_scores = mask(att_scores)
        att_weights = softmax(att_scores)
        context = mat_mul(att_weights, value)
        return context

'''attention = SingleHeadCasualAttention(inputs.shape[1], inputs.shape[1])
out = attention(inputs)
print(out)'''

class MultiHeadAttention:
    def __init__(self, emd_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = emd_dim // num_heads

        self.q_W = Linear(emd_dim, emd_dim)
        self.k_W = Linear(emd_dim, emd_dim)
        self.v_W = Linear(emd_dim, emd_dim)

        self.out_proj = Linear(emd_dim, emd_dim)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # [batch, seq_len, emb_dim]
        batch, seq_len, emb_dim = x.shape

        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        # reshape - [b, s, e_d] -> [b, s, h, h_d]

        '''
        batch
            |      
            s - head 1 - [0, 0, 0, 0 ,0, 0], [0, ]
                       - [0, 0, 0, 0 ,0, 0]
        
        d1. d2.     d3.      d4
        b - token - head 1 - [0, 0, 0, 0, 0, 0]
                  - head 2 - [0, 0, 0, 0, 0, 0]
                  - head 3 - [0, 0, 0, 0, 0, 0]
                  - head 4 - [0, 0, 0, 0, 0, 0]

        
        '''
        queries = reshape(query, (batch, seq_len, self.num_heads, self.head_dim))
        keys = reshape(key, (batch, seq_len, self.num_heads, self.head_dim))
        value = reshape(value, (batch, seq_len, self.num_heads, self.head_dim))

        # TODO - we can remove these and go straight to mat mul
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        value = value.transpose(1, 2)
        
        att_scores = mat_mul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        att_scores = mask(att_scores)
        att_weights = softmax(att_scores)
        context = mat_mul(att_weights, value)

        # TODO - we can remove these and go straight to mat mul
        context = context.transpose(1, 2)
        combined_heads_context = reshape(context, (batch, seq_len, emb_dim))

        return self.out_proj(combined_heads_context)

'''emb_dim = 24
inputs = torch.torch.randn((1, 8, emb_dim))
attention = MultiHeadAttention(emb_dim, num_heads=4)
out = attention(inputs)
print(out)'''

class SlidingWindowAttention:
    def __init__(self, emd_dim, num_heads, window_size):
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = emd_dim // num_heads

        self.q_W = Linear(emd_dim, emd_dim)
        self.k_W = Linear(emd_dim, emd_dim)
        self.v_W = Linear(emd_dim, emd_dim)

        self.out_proj = Linear(emd_dim, emd_dim)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # [batch, seq_len, emb_dim]
        batch, seq_len, emb_dim = x.shape

        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        # reshape - [b, s, e_d] -> [b, s, h, h_d]

        '''
        batch
            |      
            s - head 1 - [0, 0, 0, 0 ,0, 0], [0, ]
                       - [0, 0, 0, 0 ,0, 0]
        
        d1. d2.     d3.      d4
        b - token - head 1 - [0, 0, 0, 0, 0, 0]
                  - head 2 - [0, 0, 0, 0, 0, 0]
                  - head 3 - [0, 0, 0, 0, 0, 0]
                  - head 4 - [0, 0, 0, 0, 0, 0]

        
        '''
        queries = reshape(query, (batch, seq_len, self.num_heads, self.head_dim))
        keys = reshape(key, (batch, seq_len, self.num_heads, self.head_dim))
        value = reshape(value, (batch, seq_len, self.num_heads, self.head_dim))

        # TODO - we can remove these and go straight to mat mul
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        value = value.transpose(1, 2)
        
        att_scores = mat_mul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        att_scores = mask(att_scores, self.window_size)
        att_weights = softmax(att_scores)
        context = mat_mul(att_weights, value)

        # TODO - we can remove these and go straight to mat mul
        context = context.transpose(1, 2)
        combined_heads_context = reshape(context, (batch, seq_len, emb_dim))

        return self.out_proj(combined_heads_context)

'''emb_dim = 24
inputs = torch.torch.randn((1, 8, emb_dim))
attention = SlidingWindowAttention(emb_dim, num_heads=4, window_size=4)
out = attention(inputs)
print(out)'''

class MultiHeadLatentAttention:
    def __init__(self, emb_dim, compressed_dim, num_heads):
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.comp_head_dim = compressed_dim // num_heads

        self.q_W = Linear(emb_dim, emb_dim)

        self.k_comp_W = Linear(emb_dim, compressed_dim)
        self.k_up_W = Linear(compressed_dim, emb_dim)
        self.v_comp_W = Linear(emb_dim, compressed_dim)
        self.v_up_W = Linear(compressed_dim, emb_dim)

        self.out_proj = Linear(emb_dim, emb_dim)

        pass

    def forward(self, x):
        b, seq_len, emb_dim = x.shape

        query = self.q_W(x)

        # latent compression
        key_compressed = self.k_comp_W(x)
        value_compressed = self.v_comp_W(x)

        key = self.k_up_W(key_compressed)
        value = self.v_up_W(value_compressed)

        queries = reshape(query, (b, seq_len, self.num_heads, self.head_dim))
        keys = reshape(key, (b, seq_len, self.num_heads, self.head_dim))
        values = reshape(value, (b, seq_len, self.num_heads, self.head_dim))

        # TODO - we can remove this if keep [b, s, heads, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = mat_mul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_scores = mask(attn_scores)
        attn_weights = softmax(attn_scores)
        context = mat_mul(attn_weights, values)

        context = context.transpose(1, 2)
        context = reshape(context, (b, seq_len, emb_dim))

        return self.out_proj(context)
    
'''emb_dim = 24
inputs = torch.torch.randn((1, 8, emb_dim))
attention = MultiHeadLatentAttention(emb_dim, num_heads=4, window_size=4)
out = attention(inputs)
print(out)'''

class GroupQueryAttention:
    def __init__(self, emb_dim, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = emb_dim // num_heads

        self.q_W = Linear(emb_dim, emb_dim)
        self.k_W = Linear(emb_dim, num_kv_heads * self.head_dim)
        self.v_W = Linear(emb_dim, num_kv_heads * self.head_dim)
        self.out_proj = Linear(emb_dim, emb_dim)
    
    def __call__(self, x):
        return self.forward(x)
    
    def _repeat_kv_heads(self, x, num_groups):
        # x shape: [b, num_kv_heads, seq_len, head_dim]
        # output: [b, num_kv_heads * num_groups, seq_len, head_dim]
        b, num_kv_heads, seq_len, head_dim = x.shape
        result = Tensor((b, num_kv_heads * num_groups, seq_len, head_dim))
        
        for batch_idx in range(b):
            for kv_head_idx in range(num_kv_heads):
                for group_idx in range(num_groups):
                    output_head_idx = kv_head_idx * num_groups + group_idx
                    for seq_idx in range(seq_len):
                        for dim_idx in range(head_dim):
                            result[batch_idx][output_head_idx][seq_idx][dim_idx] = x[batch_idx][kv_head_idx][seq_idx][dim_idx]
        
        return result


    def forward(self, x):
        b, seq_len, emb_dim = x.shape
        
        query = self.q_W(x)
        key = self.k_W(x)
        value = self.v_W(x)

        queries = reshape(query, (b, seq_len, self.num_heads, self.head_dim))
        keys = reshape(key, (b, seq_len, self.num_kv_heads, self.head_dim))
        values = reshape(value, (b, seq_len, self.num_kv_heads, self.head_dim))

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Repeat K/V heads to match Q heads
        keys = self._repeat_kv_heads(keys, (self.num_heads // self.num_kv_heads))
        values = self._repeat_kv_heads(values, (self.num_heads // self.num_kv_heads))

        attn_scores = mat_mul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_scores = mask(attn_scores)
        attn_weights = softmax(attn_scores)
        context = mat_mul(attn_weights, values)
        context = context.transpose(1, 2)

        context = reshape(context, (b, seq_len, emb_dim))

        return self.out_proj(context)

emb_dim = 24
inputs = qkmx.mtrx.randn((1, 8, emb_dim))
attention = GroupQueryAttention(emb_dim, num_heads=4, num_kv_heads=2)
out = attention(inputs)
print(out)