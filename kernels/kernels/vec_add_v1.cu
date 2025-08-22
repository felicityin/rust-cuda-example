#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/cuda.h"
#include "../include/vec_add.h"

extern "C" void launch_vector_add_v1(
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

    CUDA_OK(cudaMalloc((void**)&dev_v1, n * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&dev_v2, n * sizeof(float)));
    CUDA_OK(cudaMalloc((void**)&dev_res, n * sizeof(float)));

	CUDA_OK(cudaMemcpy(dev_v1, v1, n * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(dev_v2, v2, n * sizeof(float), cudaMemcpyHostToDevice));

    size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;
    vector_add<<<grid_size, block_size>>>(dev_v1, dev_v2, dev_res, n);

    CUDA_OK(cudaMemcpy(result, dev_res, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_OK(cudaFree(dev_res));
	CUDA_OK(cudaFree(dev_v2));
    CUDA_OK(cudaFree(dev_v1));
}
