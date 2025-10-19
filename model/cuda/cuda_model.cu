#include "cuda_model.cuh"
#include <curand.h>

void cuda_gpt_init(CudaGPTModel *model, CudaGPTConfig *config) {
    model->config = *config;
    
    // Initialize CUDA handles
    CUBLAS_CHECK(cublasCreate(&model->cublas_handle));
    cudnnCreate(&model->cudnn_handle);
    
    // Allocate embedding matrices
    CUDA_CHECK(cudaMalloc(&model->d_tok_embeds, config->vocab_size * config->emb_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->d_pos_embeds, config->context_length * config->emb_dim * sizeof(float)));
    
    // Initialize embeddings with random values
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    
    curandGenerateNormal(gen, model->d_tok_embeds, config->vocab_size * config->emb_dim, 0.0f, 0.02f);
    curandGenerateNormal(gen, model->d_pos_embeds, config->context_length * config->emb_dim, 0.0f, 0.02f);
    
    curandDestroyGenerator(gen);
    
    // Allocate transformer layers
    model->attention_layers = (CudaMultiHeadAttention*)malloc(config->n_layers * sizeof(CudaMultiHeadAttention));
    model->ffn_layers = (CudaLinear*)malloc(config->n_layers * sizeof(CudaLinear));
    
    for (int i = 0; i < config->n_layers; i++) {
        cuda_attention_init(&model->attention_layers[i], config->emb_dim, config->emb_dim,
                           config->context_length, config->n_heads, config->qkv_bias, model->cublas_handle);
        cuda_linear_init(&model->ffn_layers[i], config->emb_dim, config->vocab_size, true, model->cublas_handle);
    }
    
    // Allocate layer norm parameters
    CUDA_CHECK(cudaMalloc(&model->d_ln_weight, config->emb_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->d_ln_bias, config->emb_dim * sizeof(float)));
    
    // Initialize layer norm to identity
    float *h_ln_weight = (float*)malloc(config->emb_dim * sizeof(float));
    float *h_ln_bias = (float*)malloc(config->emb_dim * sizeof(float));
    for (int i = 0; i < config->emb_dim; i++) {
        h_ln_weight[i] = 1.0f;
        h_ln_bias[i] = 0.0f;
    }
    CUDA_CHECK(cudaMemcpy(model->d_ln_weight, h_ln_weight, config->emb_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(model->d_ln_bias, h_ln_bias, config->emb_dim * sizeof(float), cudaMemcpyHostToDevice));
    free(h_ln_weight);
    free(h_ln_bias);
    
    // Allocate temporary buffers
    CUDA_CHECK(cudaMalloc(&model->d_input_embeds, config->context_length * config->emb_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->d_layer_output, config->context_length * config->emb_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&model->d_logits, config->context_length * config->vocab_size * sizeof(float)));
}

void cuda_gpt_forward(CudaGPTModel *model, const int *input_ids, int seq_len, float *output_logits) {
    // Copy input_ids to device
    int *d_input_ids;
    CUDA_CHECK(cudaMalloc(&d_input_ids, seq_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input_ids, input_ids, seq_len * sizeof(int), cudaMemcpyHostToDevice));
    
    // Token embedding lookup
    dim3 block(256);
    dim3 grid(seq_len);
    embedding_lookup_kernel<<<grid, block>>>(model->d_input_embeds, model->d_tok_embeds, 
                                            d_input_ids, seq_len, model->config.emb_dim);
    
    // Add positional embeddings
    dim3 pos_grid(seq_len);
    add_positional_embeddings_kernel<<<pos_grid, block>>>(model->d_input_embeds, model->d_pos_embeds,
                                                         seq_len, model->config.emb_dim);
    
    // Pass through transformer layers
    float *layer_input = model->d_input_embeds;
    float *layer_output = model->d_layer_output;
    
    for (int i = 0; i < model->config.n_layers; i++) {
        // Multi-head attention
        cuda_attention_forward(&model->attention_layers[i], layer_input, layer_output, 1, seq_len);
        
        // Swap buffers for next layer
        float *temp = layer_input;
        layer_input = layer_output;
        layer_output = temp;
    }
    
    // Final linear layer to get logits
    cuda_linear_forward(&model->ffn_layers[model->config.n_layers - 1], layer_input, model->d_logits, 1, seq_len);
    
    // Copy logits back to host
    CUDA_CHECK(cudaMemcpy(output_logits, model->d_logits, seq_len * model->config.vocab_size * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    cudaFree(d_input_ids);
}

int cuda_gpt_generate_token(CudaGPTModel *model, const int *input_ids, int seq_len) {
    float *logits = (float*)malloc(seq_len * model->config.vocab_size * sizeof(float));
    cuda_gpt_forward(model, input_ids, seq_len, logits);
    
    // Get logits for last token
    float *last_token_logits = logits + (seq_len - 1) * model->config.vocab_size;
    
    // Simple greedy sampling - find max logit
    int next_token = 0;
    float max_logit = last_token_logits[0];
    for (int i = 1; i < model->config.vocab_size; i++) {
        if (last_token_logits[i] > max_logit) {
            max_logit = last_token_logits[i];
            next_token = i;
        }
    }
    
    free(logits);
    return next_token;
}

void cuda_gpt_free(CudaGPTModel *model) {
    // Free embeddings
    if (model->d_tok_embeds) cudaFree(model->d_tok_embeds);
    if (model->d_pos_embeds) cudaFree(model->d_pos_embeds);
    
    // Free transformer layers
    if (model->attention_layers) {
        for (int i = 0; i < model->config.n_layers; i++) {
            cuda_attention_free(&model->attention_layers[i]);
            cuda_linear_free(&model->ffn_layers[i]);
        }
        free(model->attention_layers);
        free(model->ffn_layers);
    }
    
    // Free layer norm
    if (model->d_ln_weight) cudaFree(model->d_ln_weight);
    if (model->d_ln_bias) cudaFree(model->d_ln_bias);
    
    // Free temporary buffers
    if (model->d_input_embeds) cudaFree(model->d_input_embeds);
    if (model->d_layer_output) cudaFree(model->d_layer_output);
    if (model->d_logits) cudaFree(model->d_logits);
    
    // Destroy CUDA handles
    if (model->cublas_handle) cublasDestroy(model->cublas_handle);
    if (model->cudnn_handle) cudnnDestroy(model->cudnn_handle);
}