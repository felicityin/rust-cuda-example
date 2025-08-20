#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda.h"
#include "vec_add.h"

extern "C" const char* launch_vector_add_v3(
    const float* dev_v1,
    const float* dev_v2,
    float* dev_res,
    size_t n
) {
    if (!dev_v1 || !dev_v2 || !dev_res || n == 0) {
        fprintf(stderr, "Invalid parameters\n");
    }

    return launchKernel(vector_add, n, 0, dev_v1, dev_v2, dev_res, n);
}
