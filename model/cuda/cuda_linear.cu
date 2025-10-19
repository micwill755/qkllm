#include "cuda_linear.cuh"
#include <curand.h>

void cuda_linear_init(CudaLinear *linear, int d_in, int d_out, bool has_bias, cublasHandle_t handle) {
    linear->d_in = d_in;
    linear->d_out = d_out;
    linear->cublas_handle = handle;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&linear->d_weight, d_out * d_in * sizeof(float)));
    if (has_bias) {
        CUDA_CHECK(cudaMalloc(&linear->d_bias, d_out * sizeof(float)));
    } else {
        linear->d_bias = nullptr;
    }
    
    // Initialize weights with Xavier initialization
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    
    float scale = sqrtf(2.0f / (d_in + d_out));
    curandGenerateNormal(gen, linear->d_weight, d_out * d_in, 0.0f, scale);
    
    if (has_bias) {
        curandGenerateNormal(gen, linear->d_bias, d_out, 0.0f, 0.01f);
    }
    
    curandDestroyGenerator(gen);
}

void cuda_linear_forward(CudaLinear *linear, const float *d_input, float *d_output, 
                        int batch_size, int seq_len) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Matrix multiplication: output = input @ weight^T
    // input: (batch_size * seq_len, d_in)
    // weight: (d_out, d_in) -> weight^T: (d_in, d_out)
    // output: (batch_size * seq_len, d_out)
    CUBLAS_CHECK(cublasSgemm(linear->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            batch_size * seq_len, linear->d_out, linear->d_in,
                            &alpha,
                            d_input, batch_size * seq_len,
                            linear->d_weight, linear->d_out,
                            &beta,
                            d_output, batch_size * seq_len));
    
    // Add bias if present
    if (linear->d_bias) {
        dim3 block(256);
        dim3 grid((batch_size * seq_len * linear->d_out + block.x - 1) / block.x);
        
        auto add_bias_kernel = [] __device__ (float* output, const float* bias, int size, int d_out) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                output[idx] += bias[idx % d_out];
            }
        };
        
        add_bias_kernel<<<grid, block>>>(d_output, linear->d_bias, 
                                        batch_size * seq_len * linear->d_out, linear->d_out);
        CUDA_CHECK(cudaGetLastError());
    }
}

void cuda_linear_free(CudaLinear *linear) {
    if (linear->d_weight) cudaFree(linear->d_weight);
    if (linear->d_bias) cudaFree(linear->d_bias);
}