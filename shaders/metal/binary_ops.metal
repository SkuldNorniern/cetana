#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    result[index] = a[index] + b[index];
}

kernel void vector_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    result[index] = a[index] * b[index];
}

kernel void vector_sub(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    result[index] = a[index] - b[index];
}

kernel void vector_log(
    device const float* a [[buffer(0)]],
    device float* result [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    result[index] = log(a[index]);
}

kernel void vector_sum(
    device const float* in_array [[buffer(0)]],
    device float* out_array [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
    // Only the first thread (id == 0) will perform the sum
    if (id == 0) {
        float sum = 0.0;

        // Iterate through all elements in the input array and sum them up
        for (int i = 0; i < 5; i++) {
            sum += in_array[i];
        }

        // Store the result in the output array at the first index
        out_array[0] = sum;
    }
}
