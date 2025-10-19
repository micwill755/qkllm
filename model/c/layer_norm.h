#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

typedef struct LayerNorm {
    float *scale; 
    float *shift;
    float eps;
    int emb_dim;

    void (*init)(struct LayerNorm *ln, int emb_dim);
    float* (*forward)(struct LayerNorm *ln, float *x);
} LayerNorm;

void init_LayerNorm(LayerNorm *ln, int emb_dim) {
    ln->emb_dim = emb_dim;
    ln->eps = 0.00001f;
    
    ln->scale = (float*)malloc(emb_dim * sizeof(float));
    for (int i = 0; i < emb_dim; i++) {
        ln->scale[i] = 1.0f;
    }
    
    ln->shift = (float*)malloc(emb_dim * sizeof(float));
    for (int i = 0; i < emb_dim; i++) {
        ln->shift[i] = 0.0f;
    }
}

float* forward_LayerNorm_batch(LayerNorm *ln, float *x, int batch_size, int seq_len) {
    int total_tokens = batch_size * seq_len;
    float *output = (float*)malloc(total_tokens * ln->emb_dim * sizeof(float));
    
    // Process each token in each sequence in each batch
    for (int b = 0; b < batch_size; b++) {
        // we loop through sequence length because each token needs it own normalization
        for (int s = 0; s < seq_len; s++) {
            int token_offset = (b * seq_len + s) * ln->emb_dim;
            // gets the address of the token's first embedding value - 
            float *token_input = &x[token_offset];
            // points to where we'll write the normalized result
            float *token_output = &output[token_offset];
            
            // Calculate mean for this token
            float mean = 0.0f;
            for (int i = 0; i < ln->emb_dim; i++) {
                mean += token_input[i];
            }
            mean /= ln->emb_dim;
            
            // Calculate variance for this token
            float var = 0.0f;
            for (int i = 0; i < ln->emb_dim; i++) {
                float diff = token_input[i] - mean;
                var += diff * diff;
            }
            var /= ln->emb_dim;
            
            // Normalize this token
            for (int i = 0; i < ln->emb_dim; i++) {
                float norm_x = (token_input[i] - mean) / sqrtf(var + ln->eps);
                token_output[i] = ln->scale[i] * norm_x + ln->shift[i];
            }
        }
    }
    
    return output;
}
