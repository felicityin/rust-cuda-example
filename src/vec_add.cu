#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_OK(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)

__global__ void vector_add(
    const float* v1,
    const float* v2,
    float* result,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = v1[idx] + v2[idx];
    }
}

extern "C" void launch_vector_add(
    const float* v1,
    const float* v2,
    float* result,
    size_t n
) {
    if (!v1 || !v2 || !result || n == 0) {
        fprintf(stderr, "Invalid parameters\n");
        return;
    }

    float *dev_v1, *dev_v2, *dev_res;

    CHECK_OK(cudaMalloc((void**)&dev_v1, n * sizeof(float)));
    CHECK_OK(cudaMalloc((void**)&dev_v2, n * sizeof(float)));
    CHECK_OK(cudaMalloc((void**)&dev_res, n * sizeof(float)));

	CHECK_OK(cudaMemcpyAsync(dev_v1, v1, n * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK_OK(cudaMemcpyAsync(dev_v2, v2, n * sizeof(float), cudaMemcpyHostToDevice, stream));

    CudaStream stream;
    CHECK_OK(cudaStreamCreate(&stream));

    size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;
    vector_add<<<grid_size, block_size, 0, stream>>>(dev_v1, dev_v2, dev_res, n);

    CHECK_OK(cudaMemcpyAsync(result, dev_res, n * sizeof(float), cudaMemcpyDeviceToHost, stream));

    CHECK_OK(cudaStreamSynchronize(stream));

    CHECK_OK(cudaStreamDestroy(stream));
    CHECK_OK(cudaFree(dev_res));
	CHECK_OK(cudaFree(dev_v2));
    CHECK_OK(cudaFree(dev_v1));
}
