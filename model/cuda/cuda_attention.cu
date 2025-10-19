#include "cuda_attention.cuh"
#include <cassert>

void cuda_attention_init(CudaMultiHeadAttention *attention, int d_in, int d_out, 
                        int context_length, int num_heads, bool qkv_bias, cublasHandle_t handle) {
    assert(d_out % num_heads == 0);
    
    attention->d_in = d_in;
    attention->d_out = d_out;
    attention->num_heads = num_heads;
    attention->head_dim = d_out / num_heads;
    attention->context_length = context_length;
    attention->cublas_handle = handle;
    
    // Initialize linear layers
    cuda_linear_init(&attention->query, d_in, d_out, qkv_bias, handle);
    cuda_linear_init(&attention->key, d_in, d_out, qkv_bias, handle);
    cuda_linear_init(&attention->value, d_in, d_out, qkv_bias, handle);
    cuda_linear_init(&attention->out_proj, d_out, d_out, false, handle);
    
    // Allocate temporary buffers (max batch_size=1, max seq_len=context_length for simplicity)
    int max_tokens = context_length;
    CUDA_CHECK(cudaMalloc(&attention->d_q, max_tokens * d_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attention->d_k, max_tokens * d_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attention->d_v, max_tokens * d_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attention->d_attn_scores, num_heads * max_tokens * max_tokens * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attention->d_attn_weights, num_heads * max_tokens * max_tokens * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attention->d_context, max_tokens * d_out * sizeof(float)));
}

void cuda_attention_forward(CudaMultiHeadAttention *attention, const float *d_input, float *d_output,
                           int batch_size, int seq_len) {
    const float alpha = 1.0f, beta = 0.0f;
    const float scale = 1.0f / sqrtf(attention->head_dim);
    
    // Compute Q, K, V
    cuda_linear_forward(&attention->query, d_input, attention->d_q, batch_size, seq_len);
    cuda_linear_forward(&attention->key, d_input, attention->d_k, batch_size, seq_len);
    cuda_linear_forward(&attention->value, d_input, attention->d_v, batch_size, seq_len);
    
    // Reshape and compute attention for each head
    for (int h = 0; h < attention->num_heads; h++) {
        // Offset pointers for current head
        float *q_head = attention->d_q + h * attention->head_dim;
        float *k_head = attention->d_k + h * attention->head_dim;
        float *v_head = attention->d_v + h * attention->head_dim;
        float *scores_head = attention->d_attn_scores + h * seq_len * seq_len;
        float *weights_head = attention->d_attn_weights + h * seq_len * seq_len;
        
        // Compute attention scores: Q @ K^T
        CUBLAS_CHECK(cublasSgemm(attention->cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                seq_len, seq_len, attention->head_dim,
                                &scale,
                                k_head, attention->d_out,  // K^T (strided)
                                q_head, attention->d_out,  // Q (strided)
                                &beta,
                                scores_head, seq_len));
        
        // Apply causal mask and softmax
        dim3 block(32, 32);
        dim3 grid((seq_len + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);
        apply_causal_mask_kernel<<<grid, block>>>(scores_head, 1, 1, seq_len);
        
        // Softmax
        dim3 softmax_block(seq_len);
        dim3 softmax_grid(1);
        softmax_kernel<<<softmax_grid, softmax_block>>>(weights_head, scores_head, 1, seq_len, 1);
        
        // Compute context: weights @ V
        CUBLAS_CHECK(cublasSgemm(attention->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                attention->head_dim, seq_len, seq_len,
                                &alpha,
                                v_head, attention->d_out,  // V (strided)
                                weights_head, seq_len,
                                &beta,
                                attention->d_context + h * attention->head_dim, attention->d_out));
    }
    
    // Output projection
    cuda_linear_forward(&attention->out_proj, attention->d_context, d_output, batch_size, seq_len);
}

void cuda_attention_free(CudaMultiHeadAttention *attention) {
    cuda_linear_free(&attention->query);
    cuda_linear_free(&attention->key);
    cuda_linear_free(&attention->value);
    cuda_linear_free(&attention->out_proj);
    
    if (attention->d_q) cudaFree(attention->d_q);
    if (attention->d_k) cudaFree(attention->d_k);
    if (attention->d_v) cudaFree(attention->d_v);
    if (attention->d_attn_scores) cudaFree(attention->d_attn_scores);
    if (attention->d_attn_weights) cudaFree(attention->d_attn_weights);
    if (attention->d_context) cudaFree(attention->d_context);
}