#include <stdio.h>
#include <stdlib.h>
#include "cuda_model.cuh"
#include "../c/tokenizer.h"

void buildVocabFromFile(Tokenizer *tokenizer) {
    FILE *file = fopen("../c/the-verdict.txt", "r");
    if (!file) {
        printf("Error opening file\n");
        return;
    }

    char *vocab[10000];
    int vocab_size = 0;
    char line[1000];
    
    while (fgets(line, sizeof(line), file)) {
        char *word = strtok(line, " \n\t");
        while (word && vocab_size < 10000) {
            int exists = 0;
            for (int i = 0; i < vocab_size; i++) {
                if (strcmp(vocab[i], word) == 0) {
                    exists = 1;
                    break;
                }
            }
            if (!exists) {
                vocab[vocab_size] = (char*)malloc(strlen(word) + 1);
                strcpy(vocab[vocab_size], word);
                vocab_size++;
            }
            word = strtok(NULL, " \n\t");
        }
    }
    
    printf("Vocab size: %d\n", vocab_size);
    fclose(file);
    
    init_tokenizer(tokenizer, vocab, vocab_size);
}

int main(int argc, char *argv[]) {
    printf("Initializing CUDA GPT-2 model...\n");
    
    // Initialize CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    printf("Using CUDA device 0\n");
    
    // Build vocabulary
    Tokenizer tokenizer;
    buildVocabFromFile(&tokenizer);
    
    // Configure model
    CudaGPTConfig config = {
        .vocab_size = tokenizer.vocab_size,
        .context_length = 512,
        .emb_dim = 768,
        .n_heads = 12,
        .n_layers = 6,  // Reduced for faster training
        .drop_rate = 0.1f,
        .qkv_bias = true
    };
    
    // Initialize model
    CudaGPTModel model;
    cuda_gpt_init(&model, &config);
    printf("Model initialized with %d parameters\n", 
           config.vocab_size * config.emb_dim + config.context_length * config.emb_dim);
    
    // Test inference
    const char *test_text = "The quick brown fox";
    printf("Testing inference with: \"%s\"\n", test_text);
    
    // Simple tokenization (just use first few vocab words as tokens)
    int input_ids[] = {0, 1, 2, 3};  // Dummy tokens
    int seq_len = 4;
    
    // Generate next token
    int next_token = cuda_gpt_generate_token(&model, input_ids, seq_len);
    printf("Generated next token: %d\n", next_token);
    
    // Cleanup
    cuda_gpt_free(&model);
    printf("Model cleanup complete\n");
    
    return 0;
}