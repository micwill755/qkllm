#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>

//the lienar layer. 
typedef struct Linear {
    float *weight;  // Shape: (features_out, features_in) stored as 1D array
    float *bias;    // Shape: (features_out,)
    int d_in;
    int d_out;
} Linear;

// Helper: access weight[row][col] as weight[row * features_in + col]
float get_weight(Linear *linear, int row, int col) {
    return linear->weight[row * linear->d_in + col];
}

void init_linear(Linear *linear, int d_in, int d_out, bool has_bias) {
    linear->d_in = d_in;
    linear->d_out = d_out;
    // d_out is rows, d_in is columns
    linear->weight = (float*)malloc(d_out * d_in * sizeof(float));
    linear->bias = has_bias ? (float*)malloc(d_out * sizeof(float)) : NULL;

    // Initialize weight matrix (features_out x features_in)
    for (int i = 0; i < d_out * d_in; i++) {
        linear->weight[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    if (has_bias) {
        for (int i = 0; i < d_out; i++) {
            linear->bias[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

float* forward_linear(Linear *linear, float *x, int batch_size, int seq_len) {
    // x: input vector of size d_in
    // weight: matrix of size (d_out, d_in)
    // output: vector of size d_out
    float *output = (float*)malloc(linear->d_out * sizeof(float));
    
    // Matrix multiplication: output = x @ weight.T + bias
    for (int i = 0; i < linear->d_out; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < linear->d_in; j++) {
            // we do the matrix transpose implicitly:
            output[i] += x[j] * linear->weight[i * linear->d_in + j];
        }
        if (linear->bias) {
            output[i] += linear->bias[i];
        }
    }
    return output;
}