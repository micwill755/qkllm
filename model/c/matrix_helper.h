#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

float* transpose_matrix(float* m, int dim1, int dim2) {
    float* out = (float*) (malloc) (dim2 * dim1 * sizeof(float));
    // row and cols are from input m shape
    for (int row = 0; row < dim1; row++) {
        for (int col = 0; col < dim2; col++) {
            out[col * dim1 + row] = m[row * dim2 + col];
        }   
    }
    return out;
}

float* matrix_multiplication(float* m, int dim1, int dim2) {
    float* out = (float*) (malloc) (dim2 * dim1 * sizeof(float));
    // row and cols are from input m shape
    for (int row = 0; row < dim1; row++) {
        for (int col = 0; col < dim2; col++) {
            out[col * dim1 + row] = m[row * dim2 + col];
        }   
    }
    return out;
}