use super::{CudaError, launch::{LaunchConfig, Dim3}, stream::cudaStream_t};
use std::ptr::null_mut;
use std::marker::PhantomData;

#[link(name = "cuda")]
extern "C" {
    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> i32;
    fn cudaMemset(ptr: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum cudaMemcpyKind {
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

const CUDA_SUCCESS: i32 = 0;

/// A buffer allocated in CUDA device memory.
#[derive(Debug)]
pub struct CudaBuffer {
    ptr: *mut f32,
    size: usize,
    allocated_bytes: usize,
    pub(crate) _marker: PhantomData<*mut ()>, // Make _marker accessible within the crate
}

// Explicitly implement Send and Sync
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Allocates a new buffer of specified size on the CUDA device.
    ///
    /// # Arguments
    /// * `size` - The number of f32 elements to allocate
    ///
    /// # Returns
    /// A new CudaBuffer or an error if allocation fails
    pub fn new(size: usize) -> Result<Self, CudaError> {
        let bytes = size * std::mem::size_of::<f32>();
        let mut ptr: *mut f32 = null_mut();
        
        unsafe {
            let result = cudaMalloc(
                &mut ptr as *mut *mut f32 as *mut *mut std::ffi::c_void,
                bytes,
            );
            
            if result != CUDA_SUCCESS {
                return Err(CudaError::MemoryAllocationFailed(
                    format!("Failed to allocate CUDA memory: error code {}", result)
                ));
            }
            
            // Initialize to zeros - best practice for GPU memory
            let result = cudaMemset(ptr as *mut std::ffi::c_void, 0, bytes);
            if result != CUDA_SUCCESS {
                // Free the previously allocated memory to avoid leaks
                cudaFree(ptr as *mut std::ffi::c_void);
                return Err(CudaError::Other(
                    format!("Failed to initialize CUDA memory: error code {}", result)
                ));
            }
        }
        
        Ok(CudaBuffer { 
            ptr, 
            size,
            allocated_bytes: bytes,
            _marker: PhantomData,
        })
    }
    
    /// Returns the number of elements in the buffer
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Returns the total allocated size in bytes
    pub fn bytes(&self) -> usize {
        self.allocated_bytes
    }
    
    /// Returns raw pointer to the buffer data.
    ///
    /// # Safety
    /// This function provides direct access to the underlying memory.
    /// The caller must ensure that the device memory remains valid
    /// and is accessed in a way that respects CUDA's memory model.
    pub unsafe fn as_ptr(&self) -> *const f32 {
        self.ptr as *const f32
    }
    
    /// Returns mutable raw pointer to the buffer data.
    ///
    /// # Safety
    /// This function provides direct mutable access to the underlying memory.
    /// The caller must ensure that the device memory remains valid and is
    /// accessed in a way that respects CUDA's memory model and avoids data races.
    pub unsafe fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    /// Copies data from host to device buffer.
    ///
    /// # Arguments
    /// * `data` - The host data to copy
    ///
    /// # Returns
    /// Ok(()) or an error if the copy fails
    pub fn copy_from_host(&mut self, data: &[f32]) -> Result<(), CudaError> {
        if data.len() > self.size {
            return Err(CudaError::InvalidValue);
        }
        
        let bytes_to_copy = std::mem::size_of::<f32>() * data.len();
        
        unsafe {
            let result = cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                bytes_to_copy,
                cudaMemcpyKind::HostToDevice,
            );
            if result != CUDA_SUCCESS {
                return Err(CudaError::Other(format!(
                    "Failed to copy data to device: error code {}", 
                    result
                )));
            }
        }
        Ok(())
    }

    /// Copies data from device buffer to host.
    ///
    /// # Arguments
    /// * `data` - The host buffer to copy data into
    ///
    /// # Returns
    /// Ok(()) or an error if the copy fails
    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<(), CudaError> {
        if data.len() > self.size {
            return Err(CudaError::InvalidValue);
        }
        
        let bytes_to_copy = std::mem::size_of::<f32>() * data.len();
        
        unsafe {
            let result = cudaMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                bytes_to_copy,
                cudaMemcpyKind::DeviceToHost,
            );
            if result != CUDA_SUCCESS {
                return Err(CudaError::Other(format!(
                    "Failed to copy data from device: error code {}", 
                    result
                )));
            }
        }
        Ok(())
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr as *mut std::ffi::c_void);
        }
    }
}

/// Create an optimized launch configuration for a given number of elements
pub fn create_launch_config(size: usize, stream: cudaStream_t) -> LaunchConfig {
    const THREADS_PER_BLOCK: u32 = 256;
    let blocks = ((size as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    LaunchConfig::new(
        Dim3::new(blocks, 1, 1), 
        Dim3::new(THREADS_PER_BLOCK, 1, 1),
        0,
        stream
    )
}

/// Launch a binary vector operation kernel
unsafe fn launch_binary_op<F>(
    a: &CudaBuffer, 
    b: &CudaBuffer, 
    result: &mut CudaBuffer,
    stream: cudaStream_t,
    kernel_function: F
) -> Result<(), CudaError> 
where 
    F: FnOnce(*mut f32, *const f32, *const f32, i32, u32, u32, u32, u32, u32, u32, u32, cudaStream_t) -> i32
{
    if a.size() != b.size() || a.size() != result.size() {
        return Err(CudaError::InvalidValue);
    }
    
    let config = create_launch_config(a.size(), stream);
    
    let result_code = kernel_function(
        result.as_mut_ptr(),
        a.as_ptr(),
        b.as_ptr(),
        a.size() as i32,
        config.grid_dim.x, config.grid_dim.y, config.grid_dim.z,
        config.block_dim.x, config.block_dim.y, config.block_dim.z,
        config.shared_mem_bytes,
        stream
    );
    
    if result_code != CUDA_SUCCESS {
        return Err(CudaError::KernelLaunchFailed(
            "Binary operation kernel launch failed".into()
        ));
    }
    
    // Only synchronize if using the null stream
    if stream.is_null() {
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into()
            ));
        }
    }
    
    Ok(())
}

/// Launch a unary vector operation kernel
unsafe fn launch_unary_op<F>(
    input: &CudaBuffer, 
    result: &mut CudaBuffer,
    stream: cudaStream_t,
    kernel_function: F
) -> Result<(), CudaError> 
where 
    F: FnOnce(*mut f32, *const f32, i32, u32, u32, u32, u32, u32, u32, u32, cudaStream_t) -> i32
{
    if input.size() != result.size() {
        return Err(CudaError::InvalidValue);
    }
    
    let config = create_launch_config(input.size(), stream);
    
    let result_code = kernel_function(
        result.as_mut_ptr(),
        input.as_ptr(),
        input.size() as i32,
        config.grid_dim.x, config.grid_dim.y, config.grid_dim.z,
        config.block_dim.x, config.block_dim.y, config.block_dim.z,
        config.shared_mem_bytes,
        stream
    );
    
    if result_code != CUDA_SUCCESS {
        return Err(CudaError::KernelLaunchFailed(
            "Unary operation kernel launch failed".into()
        ));
    }
    
    // Only synchronize if using the null stream
    if stream.is_null() {
        let sync_result = cudaDeviceSynchronize();
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::Synchronization(
                "Failed to synchronize device".into()
            ));
        }
    }
    
    Ok(())
}

// Define external kernel function signatures
extern "C" {
    fn launch_vector_add_kernel(
        result: *mut f32, a: *const f32, b: *const f32, n: i32,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32, stream: cudaStream_t
    ) -> i32;
    
    fn launch_vector_multiply_kernel(
        result: *mut f32, a: *const f32, b: *const f32, n: i32,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32, stream: cudaStream_t
    ) -> i32;
    
    fn launch_vector_subtract_kernel(
        result: *mut f32, a: *const f32, b: *const f32, n: i32,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32, stream: cudaStream_t
    ) -> i32;
    
    fn launch_vector_divide_kernel(
        result: *mut f32, a: *const f32, b: *const f32, n: i32,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32, stream: cudaStream_t
    ) -> i32;
    
    fn launch_vector_exp_kernel(
        result: *mut f32, input: *const f32, n: i32,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32, stream: cudaStream_t
    ) -> i32;
    
    fn launch_vector_log_kernel(
        result: *mut f32, input: *const f32, n: i32,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32, stream: cudaStream_t
    ) -> i32;
    
    fn launch_vector_sqrt_kernel(
        result: *mut f32, input: *const f32, n: i32,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32, stream: cudaStream_t
    ) -> i32;
    
    fn launch_vector_pow_kernel(
        result: *mut f32, input: *const f32, power: f32, n: i32,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32, stream: cudaStream_t
    ) -> i32;
    
    fn launch_matrix_multiply_kernel(
        result: *mut f32, a: *const f32, b: *const f32, 
        m: i32, n: i32, k: i32,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32, stream: cudaStream_t
    ) -> i32;
    
    fn launch_vector_reduce_sum_kernel(
        result: *mut f32, input: *const f32, n: i32,
        grid_dim_x: u32, grid_dim_y: u32, grid_dim_z: u32,
        block_dim_x: u32, block_dim_y: u32, block_dim_z: u32,
        shared_mem_bytes: u32, stream: cudaStream_t
    ) -> i32;
}

/// Perform vector addition: result = a + b
pub fn vector_add(
    a: &CudaBuffer,
    b: &CudaBuffer,
    result: &mut CudaBuffer,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    unsafe {
        launch_binary_op(a, b, result, stream, |r, a, b, n, gx, gy, gz, bx, by, bz, sm, s| {
            launch_vector_add_kernel(r, a, b, n, gx, gy, gz, bx, by, bz, sm, s)
        })
    }
}

/// Perform vector multiplication: result = a * b
pub fn vector_multiply(
    a: &CudaBuffer,
    b: &CudaBuffer,
    result: &mut CudaBuffer,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    unsafe {
        launch_binary_op(a, b, result, stream, |r, a, b, n, gx, gy, gz, bx, by, bz, sm, s| {
            launch_vector_multiply_kernel(r, a, b, n, gx, gy, gz, bx, by, bz, sm, s)
        })
    }
}

/// Perform vector subtraction: result = a - b
pub fn vector_subtract(
    a: &CudaBuffer,
    b: &CudaBuffer,
    result: &mut CudaBuffer,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    unsafe {
        launch_binary_op(a, b, result, stream, |r, a, b, n, gx, gy, gz, bx, by, bz, sm, s| {
            launch_vector_subtract_kernel(r, a, b, n, gx, gy, gz, bx, by, bz, sm, s)
        })
    }
}

/// Perform vector division: result = a / b
pub fn vector_divide(
    a: &CudaBuffer,
    b: &CudaBuffer,
    result: &mut CudaBuffer,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    unsafe {
        launch_binary_op(a, b, result, stream, |r, a, b, n, gx, gy, gz, bx, by, bz, sm, s| {
            launch_vector_divide_kernel(r, a, b, n, gx, gy, gz, bx, by, bz, sm, s)
        })
    }
}

/// Compute element-wise exponential: result = exp(input)
pub fn vector_exp(
    input: &CudaBuffer,
    result: &mut CudaBuffer,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    unsafe {
        launch_unary_op(input, result, stream, |r, a, n, gx, gy, gz, bx, by, bz, sm, s| {
            launch_vector_exp_kernel(r, a, n, gx, gy, gz, bx, by, bz, sm, s)
        })
    }
}

/// Compute element-wise logarithm: result = log(input)
pub fn vector_log(
    input: &CudaBuffer,
    result: &mut CudaBuffer,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    unsafe {
        launch_unary_op(input, result, stream, |r, a, n, gx, gy, gz, bx, by, bz, sm, s| {
            launch_vector_log_kernel(r, a, n, gx, gy, gz, bx, by, bz, sm, s)
        })
    }
}

/// Compute element-wise square root: result = sqrt(input)
pub fn vector_sqrt(
    input: &CudaBuffer,
    result: &mut CudaBuffer,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    unsafe {
        launch_unary_op(input, result, stream, |r, a, n, gx, gy, gz, bx, by, bz, sm, s| {
            launch_vector_sqrt_kernel(r, a, n, gx, gy, gz, bx, by, bz, sm, s)
        })
    }
}

/// Compute element-wise power: result = input^power
pub fn vector_pow(
    input: &CudaBuffer,
    power: f32,
    result: &mut CudaBuffer,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    if input.size() != result.size() {
        return Err(CudaError::InvalidValue);
    }
    
    unsafe {
        let config = create_launch_config(input.size(), stream);
        
        let result_code = launch_vector_pow_kernel(
            result.as_mut_ptr(),
            input.as_ptr(),
            power,
            input.size() as i32,
            config.grid_dim.x, config.grid_dim.y, config.grid_dim.z,
            config.block_dim.x, config.block_dim.y, config.block_dim.z,
            config.shared_mem_bytes,
            stream
        );
        
        if result_code != CUDA_SUCCESS {
            return Err(CudaError::KernelLaunchFailed(
                "Power operation kernel launch failed".into()
            ));
        }
        
        // Only synchronize if using the null stream
        if stream.is_null() {
            let sync_result = cudaDeviceSynchronize();
            if sync_result != CUDA_SUCCESS {
                return Err(CudaError::Synchronization(
                    "Failed to synchronize device".into()
                ));
            }
        }
    }
    
    Ok(())
}

/// Perform matrix multiplication: result = a * b
/// where a is m x n, b is n x k, and result is m x k
pub fn matrix_multiply(
    a: &CudaBuffer,
    b: &CudaBuffer,
    result: &mut CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    // Validate dimensions
    if a.size() != m * n || b.size() != n * k || result.size() != m * k {
        return Err(CudaError::InvalidValue);
    }
    
    unsafe {
        // For matrix multiplication, we use a 2D grid with 16x16 blocks
        let block_dim = Dim3::new(16, 16, 1);
        let grid_dim = Dim3::new(
            (k as u32 + block_dim.x - 1) / block_dim.x,
            (m as u32 + block_dim.y - 1) / block_dim.y,
            1
        );
        
        let config = LaunchConfig::new(grid_dim, block_dim, 0, stream);
        
        let result_code = launch_matrix_multiply_kernel(
            result.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            m as i32, n as i32, k as i32,
            config.grid_dim.x, config.grid_dim.y, config.grid_dim.z,
            config.block_dim.x, config.block_dim.y, config.block_dim.z,
            config.shared_mem_bytes,
            config.stream
        );
        
        if result_code != CUDA_SUCCESS {
            return Err(CudaError::KernelLaunchFailed(
                "Matrix multiplication kernel launch failed".into()
            ));
        }
        
        // Only synchronize if using the null stream
        if stream.is_null() {
            let sync_result = cudaDeviceSynchronize();
            if sync_result != CUDA_SUCCESS {
                return Err(CudaError::Synchronization(
                    "Failed to synchronize device".into()
                ));
            }
        }
    }
    
    Ok(())
}

/// Compute the sum of all elements in the input vector
pub fn vector_reduce_sum(
    input: &CudaBuffer,
    result: &mut CudaBuffer,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    if result.size() < 1 {
        return Err(CudaError::InvalidValue);
    }
    
    unsafe {
        // For reduction, we need to determine block size and shared memory
        let block_size: u32 = 256;
        let shared_mem_size = block_size as u32 * std::mem::size_of::<f32>() as u32;
        let grid_size = (input.size() as u32 + block_size - 1) / block_size;
        
        let config = LaunchConfig::new(
            Dim3::new(grid_size, 1, 1),
            Dim3::new(block_size, 1, 1),
            shared_mem_size,
            stream
        );
        
        let result_code = launch_vector_reduce_sum_kernel(
            result.as_mut_ptr(),
            input.as_ptr(),
            input.size() as i32,
            config.grid_dim.x, config.grid_dim.y, config.grid_dim.z,
            config.block_dim.x, config.block_dim.y, config.block_dim.z,
            config.shared_mem_bytes,
            config.stream
        );
        
        if result_code != CUDA_SUCCESS {
            return Err(CudaError::KernelLaunchFailed(
                "Reduction kernel launch failed".into()
            ));
        }
        
        // Only synchronize if using the null stream
        if stream.is_null() {
            let sync_result = cudaDeviceSynchronize();
            if sync_result != CUDA_SUCCESS {
                return Err(CudaError::Synchronization(
                    "Failed to synchronize device".into()
                ));
            }
        }
    }
    
    Ok(())
}
