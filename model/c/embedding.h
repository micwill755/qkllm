#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>

typedef struct Embedding {
    float *weight; 
    int vocab_size;
    int emb_dim;
    
    void (* init)(struct Embedding *emb, int vocab_size, int emb_dim);
    float* (* forward)(struct Embedding *emb, int *input_ids, int num_tokens);
} Embedding;

// helper function to help print embeddings
void print_embeddings(Embedding emb, int row_visible, int col_visible) {
    printf("Number of embedding: %d, dimensions %d([\n", emb.vocab_size, emb.emb_dim);

    if (row_visible > emb.vocab_size) {
        row_visible = emb.vocab_size;
    }

    if (col_visible > emb.emb_dim) {
        col_visible = emb.emb_dim;
    }

    for (int i = 0; i < row_visible; i++) {
        printf("  [");
        for (int j = 0; j < col_visible; j++) {
            printf("%f", emb.weight[i * emb.emb_dim + j]);
            if (j < emb.emb_dim - 1) printf(", ");
        }
        printf("...]%s\n", i < emb.vocab_size - 1 ? "," : "");
    }
    printf("])\n");
}

void embedding_init(Embedding *emb, int vocab_size, int emb_dim) { 
    emb->vocab_size = vocab_size;
    emb->emb_dim = emb_dim;

    // Allocate weight matrix (vocab_size x emb_dim)
    emb->weight = (float*)malloc(vocab_size * emb_dim * sizeof(float));
    // Initialize with random values 
    for (int i = 0; i < vocab_size * emb_dim; i++) {
        emb->weight[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random [-1, 1]
    }
}

float* embedding_forward(Embedding *emb, int *input_ids, int num_tokens) {
    float *output = (float*)malloc(num_tokens * emb->emb_dim * sizeof(float));
    
    for (int i = 0; i < num_tokens; i++) {
        int token_id = input_ids[i];
        // Copy embedding for this token
        for (int j = 0; j < emb->emb_dim; j++) {
            output[i * emb->emb_dim + j] = emb->weight[token_id * emb->emb_dim + j];
        }
    }
    
    return output;
}
