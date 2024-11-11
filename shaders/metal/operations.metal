#include <metal_stdlib>
using namespace metal;

kernel void vector_relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    output[index] = max(0.0f, input[index]);
} 