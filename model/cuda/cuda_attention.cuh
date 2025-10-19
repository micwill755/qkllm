#ifndef CUDA_ATTENTION_CUH
#define CUDA_ATTENTION_CUH

#include "cuda_linear.cuh"

typedef struct {
    int d_in;
    int d_out;
    int num_heads;
    int head_dim;
    int context_length;
    
    CudaLinear query;
    CudaLinear key;
    CudaLinear value;
    CudaLinear out_proj;
    
    // Temporary GPU buffers
    float *d_q, *d_k, *d_v;
    float *d_attn_scores;
    float *d_attn_weights;
    float *d_context;
    
    cublasHandle_t cublas_handle;
} CudaMultiHeadAttention;

// Initialize CUDA multi-head attention
void cuda_attention_init(CudaMultiHeadAttention *attention, int d_in, int d_out, 
                        int context_length, int num_heads, bool qkv_bias, cublasHandle_t handle);

// Forward pass
void cuda_attention_forward(CudaMultiHeadAttention *attention, const float *d_input, float *d_output,
                           int batch_size, int seq_len);

// Cleanup
void cuda_attention_free(CudaMultiHeadAttention *attention);

#endif