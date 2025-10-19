#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "tokenizer.h"
#include "model.h"

// this function will read text from a file and build a vocabulary for the tokenizer
void buildVocabFromFile(Tokenizer *tokenizer){
    FILE *file = fopen("the-verdict.txt", "r");
    if (!file) {
        printf("Error opening file\n");
        return;
    }

    char *vocab[10000];
    int vocab_size = 0;
    char line[1000];
    
    // Build vocab from file
    while (fgets(line, sizeof(line), file)) {
        char *word = strtok(line, " \n\t");
        while (word && vocab_size < 10000) {
            // Check if word already exists
            int exists = 0;
            for (int i = 0; i < vocab_size; i++) {
                if (strcmp(vocab[i], word) == 0) {
                    exists = 1;
                    break;
                }
            }
            if (!exists) {
                vocab[vocab_size] = malloc(strlen(word) + 1);
                strcpy(vocab[vocab_size], word);
                vocab_size++;
            }
            word = strtok(NULL, " \n\t");
        }
    }
    
    printf("Vocab size: %d\n", vocab_size);
    fclose(file);

    // initizlize tokenizer from file contents
    init_tokenizer(tokenizer, vocab, vocab_size);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <vocab_file>\n", argv[0]);
        return 1;
    }

    GPTConfig config = {
        .vocab_size = 256,      // Will be updated after building vocab
        .context_length = 512,   // Max sequence length
        .emb_dim = 768,         // Embedding dimension
        .n_heads = 12,          // Number of attention heads
        .n_layers = 12,         // Number of transformer layers
        .drop_rate = 0.1f,      // Dropout rate
        .qkv_bias = true        // Use bias in attention
    };

    // step 1: read in file contents to build our vocab and init tokenizer
    Tokenizer tokenizer;
    buildVocabFromFile(&tokenizer);
    
    // TEMP: while we load from file - update config with actual vocab size 
    config.vocab_size = tokenizer.vocab_size;

    GPTModel model;
    GPTModel_init(&model, &config);
}