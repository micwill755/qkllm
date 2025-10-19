#include <stdint.h>
#include <ctype.h>
#include <assert.h>

typedef struct Tokenizer{
    char **vocab;     // vocabulary strings
    int *token_ids;   // corresponding token IDs
    int vocab_size;   // number of tokens

    
} Tokenizer;

#include <string.h>
#include <stdlib.h>

void init_tokenizer(Tokenizer *t, char **vocab, int vocab_size) {
    t->vocab = vocab;
    t->vocab_size = vocab_size;
    t->token_ids = malloc(vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) {
        t->token_ids[i] = i;
    }
}

int tokenize(Tokenizer *t, char *text, int *tokens, int max_tokens) {
    int token_count = 0;
    char *word = strtok(text, " ");
    
    while (word && token_count < max_tokens) {
        int token_id = -1;
        for (int i = 0; i < t->vocab_size; i++) {
            if (strcmp(word, t->vocab[i]) == 0) {
                token_id = i;
                break;
            }
        }
        tokens[token_count++] = (token_id >= 0) ? token_id : 0;
        word = strtok(NULL, " ");
    }
    return token_count;
}