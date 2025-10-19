/*
 * NCCL C Example - Multi-GPU AllReduce
 * Compile: nvcc -o nccl_example nccl_example.c -lnccl
 */

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",      \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",      \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[]) {
    int nDevs;
    CUDACHECK(cudaGetDeviceCount(&nDevs));
    printf("Number of GPUs: %d\n", nDevs);

    // Allocate and initialize host data
    size_t size = 32 * 1024 * 1024; // 32M elements
    float* hostData = (float*)malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++) {
        hostData[i] = 1.0f;
    }

    // NCCL communicators and CUDA streams
    ncclComm_t comms[nDevs];
    cudaStream_t streams[nDevs];
    
    // Device buffers
    float** sendbuff = (float**)malloc(nDevs * sizeof(float*));
    float** recvbuff = (float**)malloc(nDevs * sizeof(float*));

    // Initialize NCCL
    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));

    // Create communicators
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDevs; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(&comms[i], nDevs, id, i));
    }
    NCCLCHECK(ncclGroupEnd());

    // Allocate device memory and create streams
    for (int i = 0; i < nDevs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(&sendbuff[i], size * sizeof(float)));
        CUDACHECK(cudaMalloc(&recvbuff[i], size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(&streams[i]));
        
        // Initialize device data (each GPU gets different values)
        CUDACHECK(cudaMemcpy(sendbuff[i], hostData, size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Scale by rank for demonstration
        float scale = (float)(i + 1);
        cudaMemcpy(&scale, &scale, sizeof(float), cudaMemcpyHostToDevice);
        // Simple kernel would go here to scale the data
    }

    printf("Starting AllReduce operation...\n");
    
    // Perform AllReduce
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDevs; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    // Synchronize streams
    for (int i = 0; i < nDevs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    printf("AllReduce completed successfully!\n");

    // Verify results (copy back first few elements)
    float* result = (float*)malloc(10 * sizeof(float));
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy(result, recvbuff[0], 10 * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("First 10 elements after AllReduce: ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", result[i]);
    }
    printf("\n");

    // Cleanup
    for (int i = 0; i < nDevs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    free(hostData);
    free(result);
    free(sendbuff);
    free(recvbuff);

    printf("NCCL example completed successfully!\n");
    return 0;
}