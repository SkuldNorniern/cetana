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

extern "C" __global__ void reduce_sum_kernel(float* output, const float* input, int n) {
    extern __shared__ float sdata[];
    
    // Each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory, handling out-of-bounds
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

extern "C" __global__ void reduce_blocks_kernel(float* output, const float* input, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    // Load data directly to shared memory
    sdata[tid] = (tid < n) ? input[tid] : 0.0f;
    __syncthreads();
    
    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result to output
    if (tid == 0) *output = sdata[0];
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
    float* result, const float* input, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, cudaStream_t stream
) {
    if (n <= 0) {
        *result = 0.0f;
        return 0;
    }
    
    // Limit block size to avoid exceeding shared memory
    const unsigned int max_block_size = 1024;
    unsigned int block_size = block_dim_x * block_dim_y * block_dim_z;
    if (block_size > max_block_size) block_size = max_block_size;
    
    // Calculate grid size to cover the entire input
    unsigned int grid_size = (n + block_size - 1) / block_size;
    
    // Allocate temporary storage for block results
    float* d_block_sums = nullptr;
    cudaError_t error = cudaMalloc(&d_block_sums, grid_size * sizeof(float));
    if (error != cudaSuccess) return (int)error;
    
    // Launch the first kernel to reduce blocks
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(grid_size, 1, 1);
    unsigned int smem_size = block_size * sizeof(float);
    
    reduce_sum_kernel<<<dimGrid, dimBlock, smem_size, stream>>>(
        d_block_sums, input, n
    );
    
    // If grid_size is 1, we can directly copy the result
    if (grid_size == 1) {
        error = cudaMemcpyAsync(result, d_block_sums, sizeof(float), 
                               cudaMemcpyDeviceToDevice, stream);
        cudaFree(d_block_sums);
        return (int)error;
    }
    
    // Otherwise, we need another reduction step
    // Reduce the block sums
    reduce_blocks_kernel<<<1, 1024, 1024 * sizeof(float), stream>>>(
        result, d_block_sums, grid_size
    );
    
    // Free temporary storage
    cudaFree(d_block_sums);
    
    return 0;
}