#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

// CUDA kernel declarations
__global__ void embedding_lookup_kernel(float* output, const float* weight, const int* input_ids, 
                                       int num_tokens, int emb_dim);

__global__ void add_positional_embeddings_kernel(float* embeddings, const float* pos_embeddings,
                                                int seq_len, int emb_dim);

__global__ void softmax_kernel(float* output, const float* input, int batch_size, int seq_len, int num_heads);

__global__ void apply_causal_mask_kernel(float* attn_scores, int batch_size, int num_heads, int seq_len);

__global__ void transpose_kernel(float* output, const float* input, int rows, int cols);

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#endif