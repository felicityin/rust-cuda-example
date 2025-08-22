#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
