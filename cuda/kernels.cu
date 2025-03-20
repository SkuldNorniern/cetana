#include "gcc13_compat.h"

extern "C" __global__ void vector_add_kernel(float *result, const float *a, const float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void vector_subtract_kernel(float *result, const float *a, const float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = a[idx] - b[idx];
    }
}

extern "C" __global__ void vector_multiply_kernel(float *result, const float *a, const float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = a[idx] * b[idx];
    }
}

extern "C" __global__ void vector_divide_kernel(float *result, const float *a, const float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = a[idx] / b[idx];
    }
}

extern "C" __global__ void vector_exp_kernel(float *result, const float *a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = expf(a[idx]);
    }
}

extern "C" __global__ void vector_log_kernel(float *result, const float *a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = logf(a[idx]);
    }
}

extern "C" __global__ void vector_pow_kernel(float *result, const float *a, float power, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = powf(a[idx], power);
    }
}

extern "C" __global__ void vector_sqrt_kernel(float *result, const float *a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = sqrtf(a[idx]);
    }
}

extern "C" __global__ void matrix_multiply_kernel(float *result, const float *a, const float *b,
                                                  int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        result[row * k + col] = sum;
    }
}

extern "C" __global__ void vector_reduce_sum_kernel(float *result, const float *a, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? a[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        result[blockIdx.x] = sdata[0];
}

// Kernel launch wrapper functions
extern "C" int launch_vector_add_kernel(
    float *result, const float *a, const float *b, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream)
{
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    
    vector_add_kernel<<<grid, block, shared_mem_bytes, (cudaStream_t)stream>>>(result, a, b, n);
    return cudaGetLastError();
}

extern "C" int launch_vector_multiply_kernel(
    float *result, const float *a, const float *b, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream)
{
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    
    vector_multiply_kernel<<<grid, block, shared_mem_bytes, (cudaStream_t)stream>>>(result, a, b, n);
    return cudaGetLastError();
}

extern "C" int launch_vector_subtract_kernel(
    float *result, const float *a, const float *b, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream)
{
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    
    vector_subtract_kernel<<<grid, block, shared_mem_bytes, (cudaStream_t)stream>>>(result, a, b, n);
    return cudaGetLastError();
}

extern "C" int launch_vector_divide_kernel(
    float *result, const float *a, const float *b, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream)
{
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    
    vector_divide_kernel<<<grid, block, shared_mem_bytes, (cudaStream_t)stream>>>(result, a, b, n);
    return cudaGetLastError();
}

extern "C" int launch_vector_exp_kernel(
    float *result, const float *a, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream)
{
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    
    vector_exp_kernel<<<grid, block, shared_mem_bytes, (cudaStream_t)stream>>>(result, a, n);
    return cudaGetLastError();
}

extern "C" int launch_vector_log_kernel(
    float *result, const float *a, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream)
{
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    
    vector_log_kernel<<<grid, block, shared_mem_bytes, (cudaStream_t)stream>>>(result, a, n);
    return cudaGetLastError();
}

extern "C" int launch_vector_sqrt_kernel(
    float *result, const float *a, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream)
{
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    
    vector_sqrt_kernel<<<grid, block, shared_mem_bytes, (cudaStream_t)stream>>>(result, a, n);
    return cudaGetLastError();
}

extern "C" int launch_vector_pow_kernel(
    float *result, const float *a, float power, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream)
{
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    
    vector_pow_kernel<<<grid, block, shared_mem_bytes, (cudaStream_t)stream>>>(result, a, power, n);
    return cudaGetLastError();
}

extern "C" int launch_matrix_multiply_kernel(
    float *result, const float *a, const float *b, int m, int n, int k,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream)
{
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    
    matrix_multiply_kernel<<<grid, block, shared_mem_bytes, (cudaStream_t)stream>>>(result, a, b, m, n, k);
    return cudaGetLastError();
}

extern "C" int launch_vector_reduce_sum_kernel(
    float *result, const float *a, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream)
{
    dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);
    dim3 block(block_dim_x, block_dim_y, block_dim_z);
    
    vector_reduce_sum_kernel<<<grid, block, shared_mem_bytes, (cudaStream_t)stream>>>(result, a, n);
    return cudaGetLastError();
}