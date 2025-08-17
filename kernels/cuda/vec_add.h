#pragma once

__global__ void vector_add(
    const float* v1,
    const float* v2,
    float* result,
    size_t n
);
