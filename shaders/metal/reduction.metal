#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint& length [[buffer(2)]],
    threadgroup float* shared_memory [[threadgroup(0)]],
    uint thread_position_in_grid [[thread_position_in_grid]],
    uint thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]])
{
    const uint tid = thread_position_in_threadgroup;
    const uint gid = thread_position_in_grid;
    
    shared_memory[tid] = (gid < length) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = threadgroup_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_memory[tid] += shared_memory[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        atomic_fetch_add_explicit((device atomic_uint*)output, 
                                as_type<uint>(shared_memory[0]), 
                                memory_order_relaxed);
    }
} 