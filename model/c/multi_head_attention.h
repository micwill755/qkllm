#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>

#include "linear.h"
#include "matrix_helper.h"

typedef struct MultiHeadAttention {
    int d_out;
    int num_heads;
    int d_in;
    int head_dim;
    int dropout;
    int context_length;

    Linear query;
    Linear key;
    Linear value;
    Linear out_proj;

    float *mask;
} MultiHeadAttention;

// helper function
// Access mask[i][j] as mask[i * context_length + j]
float get_mask_value(MultiHeadAttention *attention, int i, int j) {
    return attention->mask[i * attention->context_length + j];
}

void init_MultiHeadAttention (MultiHeadAttention *attention, int d_in, int d_out, int context_length, int dropout, int num_heads, bool qkv_bias) {
    assert(d_out % num_heads == 0 && "d_out must be divisible by n_heads");
    attention->d_in = d_in;
    attention->d_out = d_out;
    attention->num_heads = num_heads;
    attention->head_dim = d_out / num_heads; 
    attention->dropout = dropout;

    init_linear(&attention->query, d_in, d_out, qkv_bias); 
    init_linear(&attention->key, d_in, d_out, qkv_bias);  
    init_linear(&attention->value, d_in, d_out, qkv_bias); 
    init_linear(&attention->out_proj, d_in, d_out, false); 

    attention->mask = (float*)malloc(context_length * context_length * sizeof(float));

    // Create upper triangular matrix 
    for (int row = 0; row < context_length; row++) {
        for (int col = 0; col < context_length; col++) {
            attention->mask[row * context_length + col] = col < (row + 1) ? 0.0f : 1.0f;
        }
    }
}

void forward_MultiHeadAttention (MultiHeadAttention *attention, float *x, int batch_size, int seq_len) {
    // In your MultiHeadAttention
    float* keys = forward_linear(&attention->key, x, batch_size, seq_len);
    float* queries = forward_linear(&attention->query, x, batch_size, seq_len);
    float* values = forward_linear(&attention->value, x, batch_size, seq_len);


    transpose_matrix(keys, attention->d_out, attention->d_in);
}

/*def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key.forward(x)
        queries = self.W_query.forward(x)
        values = self.W_value.forward(x)

        keys = keys.reshape(b, num_tokens, self.num_heads, self.head_dim)
        values = values.reshape(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.reshape(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask[:num_tokens, :num_tokens].astype(bool)
        attn_scores = np.where(mask_bool, -np.inf, attn_scores)

        attn_weights = self.softmax(attn_scores / np.sqrt(keys.shape[-1]))
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj.forward(context_vec)

        return context_vec
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)*/