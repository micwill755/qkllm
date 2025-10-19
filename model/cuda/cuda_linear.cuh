#ifndef CUDA_LINEAR_CUH
#define CUDA_LINEAR_CUH

#include "cuda_kernels.cuh"

typedef struct {
    float *d_weight;  // Device weight matrix
    float *d_bias;    // Device bias vector
    int d_in;
    int d_out;
    cublasHandle_t cublas_handle;
} CudaLinear;

// Initialize CUDA linear layer
void cuda_linear_init(CudaLinear *linear, int d_in, int d_out, bool has_bias, cublasHandle_t handle);

// Forward pass using cuBLAS
void cuda_linear_forward(CudaLinear *linear, const float *d_input, float *d_output, 
                        int batch_size, int seq_len);

// Cleanup
void cuda_linear_free(CudaLinear *linear);

#endif