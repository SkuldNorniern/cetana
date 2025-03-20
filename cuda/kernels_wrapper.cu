// Minimal CUDA wrapper that provides kernel interfaces
#include <cuda_runtime.h>

// Function to initialize CUDA - implemented in cuda_init.cpp
extern "C" int cudaInit(unsigned int flags);

// External kernel declarations
extern "C" int launch_vector_add_kernel(
    float *result, const float *a, const float *b, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream);

extern "C" int launch_vector_multiply_kernel(
    float *result, const float *a, const float *b, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream);

extern "C" int launch_vector_subtract_kernel(
    float *result, const float *a, const float *b, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream);
    
extern "C" int launch_vector_divide_kernel(
    float *result, const float *a, const float *b, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream);
    
extern "C" int launch_vector_exp_kernel(
    float *result, const float *a, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream);
    
extern "C" int launch_vector_log_kernel(
    float *result, const float *a, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream);
    
extern "C" int launch_vector_sqrt_kernel(
    float *result, const float *a, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream);
    
extern "C" int launch_vector_pow_kernel(
    float *result, const float *a, float power, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream);
    
extern "C" int launch_matrix_multiply_kernel(
    float *result, const float *a, const float *b, int m, int n, int k,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream);
    
extern "C" int launch_vector_reduce_sum_kernel(
    float *result, const float *a, int n,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void *stream);
