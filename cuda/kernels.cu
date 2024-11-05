extern "C" __global__ void vector_sub_kernel(float *result, const float *a, const float *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = a[idx] - b[idx];
    }
}

extern "C" __global__ void vector_div_kernel(float *result, const float *a, const float *b, int n)
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