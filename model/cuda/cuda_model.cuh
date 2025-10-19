#ifndef CUDA_MODEL_CUH
#define CUDA_MODEL_CUH

#include "cuda_attention.cuh"
#include "../c/tokenizer.h"

typedef struct {
    int vocab_size;
    int context_length;
    int emb_dim;
    int n_heads;
    int n_layers;
    float drop_rate;
    bool qkv_bias;
} CudaGPTConfig;

typedef struct {
    CudaGPTConfig config;
    
    // Embeddings
    float *d_tok_embeds;    // Token embeddings
    float *d_pos_embeds;    // Positional embeddings
    
    // Transformer layers
    CudaMultiHeadAttention *attention_layers;
    CudaLinear *ffn_layers;  // Feed-forward networks
    
    // Layer normalization parameters
    float *d_ln_weight;
    float *d_ln_bias;
    
    // Temporary buffers
    float *d_input_embeds;
    float *d_layer_output;
    float *d_logits;
    
    // CUDA handles
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
} CudaGPTModel;

// Initialize CUDA GPT model
void cuda_gpt_init(CudaGPTModel *model, CudaGPTConfig *config);

// Forward pass
void cuda_gpt_forward(CudaGPTModel *model, const int *input_ids, int seq_len, float *output_logits);

// Generate next token
int cuda_gpt_generate_token(CudaGPTModel *model, const int *input_ids, int seq_len);

// Cleanup
void cuda_gpt_free(CudaGPTModel *model);

#endif