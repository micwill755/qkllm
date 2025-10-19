#include "cuda_kernels.cuh"
#include <math.h>

// Embedding lookup kernel
__global__ void embedding_lookup_kernel(float* output, const float* weight, const int* input_ids, 
                                       int num_tokens, int emb_dim) {
    int token_idx = blockIdx.x;
    int emb_idx = threadIdx.x;
    
    if (token_idx < num_tokens && emb_idx < emb_dim) {
        int token_id = input_ids[token_idx];
        output[token_idx * emb_dim + emb_idx] = weight[token_id * emb_dim + emb_idx];
    }
}

// Add positional embeddings
__global__ void add_positional_embeddings_kernel(float* embeddings, const float* pos_embeddings,
                                                int seq_len, int emb_dim) {
    int pos = blockIdx.x;
    int dim = threadIdx.x;
    
    if (pos < seq_len && dim < emb_dim) {
        embeddings[pos * emb_dim + dim] += pos_embeddings[pos * emb_dim + dim];
    }
}

// Softmax kernel with temperature scaling
__global__ void softmax_kernel(float* output, const float* input, int batch_size, int seq_len, int num_heads) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || row >= seq_len) return;
    
    int offset = batch_idx * num_heads * seq_len * seq_len + head_idx * seq_len * seq_len + row * seq_len;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int col = 0; col <= row; col++) {  // Causal mask: only look at current and previous tokens
        max_val = fmaxf(max_val, input[offset + col]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int col = 0; col <= row; col++) {
        float exp_val = expf(input[offset + col] - max_val);
        output[offset + col] = exp_val;
        sum += exp_val;
    }
    
    // Normalize and apply causal mask
    for (int col = 0; col < seq_len; col++) {
        if (col <= row) {
            output[offset + col] /= sum;
        } else {
            output[offset + col] = 0.0f;  // Causal mask
        }
    }
}

// Apply causal mask to attention scores
__global__ void apply_causal_mask_kernel(float* attn_scores, int batch_size, int num_heads, int seq_len) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int row = threadIdx.x;
    int col = threadIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || row >= seq_len || col >= seq_len) return;
    
    int idx = batch_idx * num_heads * seq_len * seq_len + head_idx * seq_len * seq_len + row * seq_len + col;
    
    if (col > row) {
        attn_scores[idx] = -INFINITY;  // Mask future tokens
    }
}

// Matrix transpose kernel
__global__ void transpose_kernel(float* output, const float* input, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}