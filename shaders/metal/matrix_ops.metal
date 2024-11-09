#include <metal_stdlib>
using namespace metal;

kernel void matrix_multiply(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    device const uint& M [[buffer(3)]],
    device const uint& N [[buffer(4)]],
    device const uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= N || gid.y >= M) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += a[gid.y * K + k] * b[k * N + gid.x];
    }
    result[gid.y * N + gid.x] = sum;
} 